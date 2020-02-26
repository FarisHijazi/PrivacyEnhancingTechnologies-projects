import argparse
import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import types

from encryption_utils import CipherLib, _string_to_bytes, _bytes_to_string
from utils import recv_msg, send_msg, AttrDict

# 256 bits = 32 bytes
# b'c37ddfe20d88021bc66a06706ac9fbdd0bb2dc0b043cf4d22dbbbcda086f0f48'
DEFAULT_KEY = _bytes_to_string(
    b'\xc3\x7d\xdf\xe2\x0d\x88\x02\x1b\xc6\x6a\x06\x70\x6a\xc9\xfb\xdd\x0b\xb2\xdc\x0b\x04\x3c\xf4\xd2\x2d\xbb\xbc\xda\x08\x6f\x0f\x48')


def send_command(args, callback=lambda sock: print("Connected", sock)):
    """connects to the server and sends the command

    :param args:   this object is similar to the one parsed from the commandline,
        contains "host" and "port" members
    :param callback(sock, respjson): a function to call when connected to the server.
        sock:   Gets passed the socket object, the socket object at this point is already connected and is ready to send or recv.
    :return the callback result
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('connecting to server...', end='')
        s.connect((args.host, args.port))  # connect
        print('\rConnection established                       ')

        # random initialization vector
        setattr(args, 'iv', secrets.token_bytes(16))

        if not hasattr(args, 'cipherfunc'):
            setattr(args, 'cipherfunc', CipherLib.none)

        ################
        # serialize args
        ################
        import copy
        s_args = copy.deepcopy(vars(args))
        for k, v in s_args.items():
            if isinstance(v, types.FunctionType):  # functions get the name passed
                s_args[k] = v.__name__
            elif isinstance(v, bytes):  # bytes get turned into strings
                s_args[k] = _bytes_to_string(v)

        s_args['cipher'] = s_args.get('cipherfunc', 'none')
        del s_args['key']  # delete key (otherwise is sent in plaintext)

        request_json = json.dumps(s_args)
        print('Sending command: "{}"'.format(request_json))

        # send the command/request json
        send_msg(s, _string_to_bytes(request_json))

        # check if server acknowledged the command
        # (if resp is included in one of the success response codes)
        resp = recv_msg(s)
        resp_json = AttrDict(json.loads(_bytes_to_string(resp)))
        if resp_json.readystate in [202]:
            res = callback(s)

            send_msg(s, b'200')  # send OK code
            print('\nTransaction complete')
            return res


def get_user_commands(parser: argparse.ArgumentParser, args=None):
    # the returned agrs object will also have a member args._line_args
    # parsing args
    line_args = ''

    if not args:
        args = parser.parse_args()
        if hasattr(args, 'function'):  # arguments passed (if first time)
            line_args = ' '.join(sys.argv[1:])
            sys.argv = [sys.argv[0]]  # clear CLI args

    if not line_args:  # no args passed
        done = False
        while not done:
            parser.print_usage()
            values_as_strings = [(v.__name__ if hasattr(v, '__name__') else str(v)) for v in args.__dict__.values()]
            args_str = dict(zip(args.__dict__.keys(), values_as_strings))

            # # pretty printing the arguments
            # print("Current arg values:")
            # import pprint
            # pprint.PrettyPrinter(indent=4).pprint(args_str)

            line_args = input('Client\n$ ')
            print()

            try:
                args = parser.parse_args(shlex.split(line_args))
                done = True  # keep trying and break when successful
            except Exception as e:
                print("ERROR:", e)

    setattr(args, '_line_args', line_args)
    return args


def exec_function(args):
    if not hasattr(args, 'function'):
        return

    return args.function(args)


def get_arg_parser():
    """
    creating the command parser object
    """
    parser = argparse.ArgumentParser(description="Connect to server")

    # act normally in the beginning (from the command line)
    parser.add_argument('--port', default=65431, type=int,
                        help='port to listen on (non-privileged ports are > 1023).'
                             'Default: 65432')
    parser.add_argument('--host', default='127.0.0.1', type=str,
                        help='hostname or ipv4 address to connect to (use ip address for consistency).'
                             'Default: "127.0.0.1"')

    parser.add_argument('-k', '--key', default=DEFAULT_KEY,  # 256 bits = 32 bytes
                        help='The key used for encryption/decryption.')

    parser.add_argument('-m', '--msg-index', default=0, type=int,
                        help='The index of the message to receive. Choose a value between [0, n-1]')

    parser.add_argument('-n', '--n-msgs', default=2,
                        help='Number of messages: 1-out-of-(n) (minimum: 2)')

    return parser


# ============ client actions =======


def get(args=None):
    def callback(conn: socket):
        # receive data
        resp = AttrDict(json.loads(_bytes_to_string(recv_msg(conn))))

        if args.file_index:
            args.filename = resp.filename
            delattr(args, 'file_index')

        if not os.path.isdir('./files'):
            os.mkdir('./files')

        filename = args.filename \
            if args.filename.startswith('files') \
            else os.path.join('files', args.filename)

        if os.path.isdir(filename):
            args.filename = os.path.join(args.filename, resp.filename)

        # === done preparing filesystem ===

        with open(filename, 'wb+') as f:
            plaintext = args.cipherfunc(data=resp.data, key=args.key, decrypt=True, iv=resp.iv)
            f.write(plaintext)
            if os.path.isfile(filename):
                subprocess.Popen(r'explorer /select,"{}"'.format(filename))

    return send_command(args, callback)


def put(args=None):
    if args.file_index:  # if access-by-fileindex, then remove attr (to prevent issues) and get filename
        delattr(args, 'file_index')
        file_index = int(args.filename)
        args.filename = ls_local(args)[file_index]

    filename = os.path.join('files', args.filename)  # prepend 'file/'

    if not os.path.isfile(filename):  # check if file exists
        print('ERROR: File "{}" doesn\'t exist'.format(filename))
        return

    def callback(conn: socket):
        ciphertext = b''
        with open(filename, 'rb') as f:
            data = f.read()
            ciphertext = args.cipherfunc(data=data, key=args.key, iv=args.iv)

        return send_msg(
            conn,
            _string_to_bytes(json.dumps({
                'filename': filename,
                'data': _bytes_to_string(ciphertext),
                'iv': _bytes_to_string(args.iv),
            }))
        )

    return send_command(args, callback)


def ls(args):
    """
    list files, either local or online (depending on --local argument)
    """
    if args.local:
        return ls_local(args, True)
    else:
        return ls_remote(args)


def ls_local(args=None, print_list=False):
    filelist = os.listdir('files/')
    if print_list:
        prettystr = '\n'.join(['\t{} | \t{}'.format(i, file)
                               for i, file in enumerate(filelist)])
        print("List of server files:\n", prettystr)
    return filelist


def ls_remote(args):
    def callback(conn: socket):
        resp = recv_msg(conn)
        filelist = json.loads(resp)
        prettystr = '\n'.join(['\t{} | \t{}'.format(i, file)
                               for i, file in enumerate(filelist)])
        print("List of server files:\n", prettystr)
        return filelist

    return send_command(args, callback)


def quit(args=None):
    exit(0)
