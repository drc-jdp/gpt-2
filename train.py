import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

import model, sample, encoder
from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

CHECKPOINT_DIR = '../storage'
SAMPLE_DIR = 'samples'


parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--encoding', type=str, default='utf-8', help='Set the encoding for reading and writing files.')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
parser.add_argument('--noise', type=float, default=0.0, help='Add noise to input training data to regularize against typos.')

parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=2, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context


def get_hparam(args):
    hparams = model.default_hparams()
    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    return hparams


def get_ckpt(args):
    # restore from the dir where model saving
    ckpt = None
    if args.restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(
            os.path.join(CHECKPOINT_DIR, args.run_name))
    return ckpt


def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    hparams = get_hparam(args)
    ckpt = get_ckpt(args)

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

    with tf.Session(config=config) as sess:
        # init training tensor
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        context_in = randomize(context, hparams, args.noise)
        output = model.model(hparams=hparams, X=context_in)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        # init val tensor
        if args.val_every > 0:
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_summary = tf.summary.scalar('val_loss', val_loss)

        # init sample tensor
        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=args.top_k,
            top_p=args.top_p)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

        # init training
        opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        if args.memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)
        summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])
        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            with open(
                    os.path.join(CHECKPOINT_DIR, args.run_name, 'log'), 'a', encoding=args.encoding) as fp:
                fp.write(f"save model {counter}\n")
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            with open(
                    os.path.join(SAMPLE_DIR, args.run_name,
                                 'samples-{}').format(counter), 'w', encoding=args.encoding) as fp:
                fp.write('\n'.join(all_text))

        def validation():
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))

            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            with open(
                    os.path.join(CHECKPOINT_DIR, args.run_name, 'log'), 'a', encoding=args.encoding) as fp:
                fp.write(
                    '[{counter} | {time:2.2f}] validation loss = {loss:.10f}\n'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_val_loss))

        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(args.batch_size)]

        print('Loading checkpoint', ckpt)
        if ckpt is not None:
            saver.restore(sess, ckpt)
        else:
            save()
            sys.exit()

        print('Loading dataset...')
        chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
        data_sampler = Sampler(chunks)
        if args.val_every > 0:
            if args.val_dataset:
                print('Loading validation dataset...')
                val_chunks = load_dataset(enc, args.val_dataset, args.combine, encoding=args.encoding)
            else:
                val_chunks = chunks
        print('dataset has', data_sampler.total_size, 'tokens')

        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            print('validation data has ', val_data_sampler.total_size, ' tokens')
            val_batches = [[val_data_sampler.sample(1024) for _ in range(args.val_batch_size)]
                           for _ in range(args.val_batch_count)]

        print('Training...')
        avg_loss = (0.0, 0.0)
        start_time = time.time()
        try:
            while True:
                if counter % args.save_every == 0:
                    save()
                    ls_c = counter - args.save_every*5
                    if ls_c > 0:
                        fn = "model-" + str(ls_c)
                        for s in [".data-00000-of-00001", ".index", ".meta"]:
                            full_str = os.path.join(CHECKPOINT_DIR, args.run_name, fn+s)
                            if os.path.isfile(full_str):
                                os.remove(full_str)
                if counter % args.sample_every == 0:
                    generate_samples()
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    validation()

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        sess.run(
                            opt_compute, feed_dict={context: sample_batch()})
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries),
                        feed_dict={context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                if counter % 10 == 0:
                    with open(
                            os.path.join(CHECKPOINT_DIR, args.run_name, 'log'), 'a', encoding=args.encoding) as fp:
                        fp.write(
                            '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}\n'
                            .format(
                                counter=counter,
                                time=time.time() - start_time,
                                loss=v_loss,
                                avg=avg_loss[0] / avg_loss[1])
                        )

                counter += 1
        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
