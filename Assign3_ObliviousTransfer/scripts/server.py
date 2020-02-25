import os
import socket
import secrets, json
import rsa
from cryptography.fernet import Fernet

# moving the import path 1 directory up (to import utils)

os.chdir(os.path.dirname(os.path.realpath(__file__)))  # move path to file dir, to access files

from server_backend import get_arg_parser, recv_next_command
from utils import recv_msg, send_msg
from encryption_utils import _bytes_to_string, _string_to_bytes

# this parser is to parse the client commands (not the commandline parser)


if __name__ == "__main__":
    print("COE449 Assignment3: 1-out-of-2 oblivious transfer"
          "\nServer side: Sender (holder of the n messages) (Alice)" +
          "\nFaris Hijazi s201578750 26-11-19" +
          "\n=======================================")
    args = get_arg_parser().parse_args()

    # the messages to choose from (assuming here a maximum of 6)
    messages = [
        '0. Big Been'
        '1. hello',
        '2. I like Trudy'
        '3. E',
        '4. DOOM is great',
        '5. DOOT'
    ]

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((args.host, args.port))
            s.listen(0)
            print("waiting for clients to connect...")
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                # try:
                #    Step 1: Alice generates an RSA key pair PK=(n,e) and SK=(d)

                pubkey, privkey = rsa.newkeys(32, poolsize=8)

                print(pubkey)
                print(privkey)

                #    generates n random  values, r_0 and r_1, and sends them to Bob along with her PK
                r_b_choices = [secrets.token_bytes(4) for i in range(args.n_msgs)]
                
                json_string = json.dumps({
                    'e': pubkey.e,
                    'n': pubkey.n,
                    'r_b_choices': list(map(_bytes_to_string, r_b_choices)),
                })
                send_msg(conn, _string_to_bytes(json_string))

                # Step 4: Alice doesn't know which of  r_0 and r_1 Bob chose.
                #         She applies them both and come up with two possible values for k:k_0=(v−x_0 )^d  mod n and k_1=(v−x_1 )^d  mod n  Eventually,
                #         one of these will be equal to  k and can be correctly decrypted by Bob (but not Alice),
                #         while the other will produce a meaningless random value that does not  reveal any  information about k
                resp4 = recv_msg(conn)
                v = _bytes_to_string(resp4)
                print('v=', v)
                print('r_b_choices=', r_b_choices)
                k_list = [privkey.unblind(v, int.from_bytes(r_b, 'big')) for r_b in r_b_choices]
                
                print('k_list=', k_list)

                # combining the message with the key: m' = m+k
                # combined_msgs = [messages[i] ^ k_list[i] for i in range(args.n_msgs)]
                print('before getting msgs')
                
                combined_msgs = [Fernet(k_list[i]).encrypt(messages[i]) for i in range(args.n_msgs)]
                print('after msg list')
                combined_msgs = ','.join(list(map(_bytes_to_string, combined_msgs))) # serialized
                print('sending combined_msgs:', combined_msgs)
                send_msg(conn, combined_msgs)

                send_msg(s, b'202')  # send DONE code

                final_client_resp = recv_msg(conn)
                if final_client_resp in [b'200']:
                    print("Transaction completed successfully")
                # except Exception as e:
                #     print("Error:", e)
                #     continue

        print("Closing connection")
