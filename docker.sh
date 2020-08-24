sudo docker stop train && sudo docker rm train
sudo docker rmi dtp-training
sudo docker build -t dtp-training . 

sudo docker run -e RESTORE_FROM=fresh -itd -v /mnt/disks/gce-containers-mounts/gce-persistent-disks/intern/save_model:/home/storage/training --name train dtp-training
# sudo docker exec -it dtp bash
