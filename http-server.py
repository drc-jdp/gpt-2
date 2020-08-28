from http.server import HTTPServer
from http.server import SimpleHTTPRequestHandler
import sys
import base64

key = ""

class AuthHandler(SimpleHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"Test\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        if self.headers['Authorization'] is None:
            self.do_AUTHHEAD()
            self.wfile.write(bytes('no auth header received', 'utf-8'))
        elif self.headers['Authorization'] == 'Basic '+key.decode('utf-8'):
            SimpleHTTPRequestHandler.do_GET(self)
        else:
            self.do_AUTHHEAD()
            self.wfile.write(self.headers['Authorization'].encode('utf-8'))
            self.wfile.write('not authenticated'.encode('utf-8'))

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        print(body)
        print(self.headers)
        ''' Present frontpage with user authentication. '''
        if self.headers['Authoorization'] == None:
            self.do_AUTHHEAD()
            self.wfile.write('no auth header received'.encode('utf-8'))
        elif self.headers['Authoorization'] == 'Basic '+key.decode('utf-8'):
            SimpleHTTPRequestHandler.do_GET(self)
        else:
            self.do_AUTHHEAD()
            self.wfile.write(self.headers['Authoorization'].encode('utf-8'))
            self.wfile.write('not authenticated'.encode('utf-8'))


if __name__ == '__main__':
    if len(sys.argv)<3:
        print ("usage HTTPBasicAuthServer.py [port] [username:password]")
        sys.exit()
    key = base64.b64encode(bytes(sys.argv[2], 'utf-8'))

    server_address = ('', int(sys.argv[1]) )
    httpd = HTTPServer(server_address, AuthHandler)
    httpd.serve_forever()
