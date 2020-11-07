import urllib.request

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()
response = opener.open('http://httpbin.org/user-agent')


src_filename = file_name
dest_filename = 'output.wav'

process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
if process.returncode != 0:
    raise Exception("Something went wrong")