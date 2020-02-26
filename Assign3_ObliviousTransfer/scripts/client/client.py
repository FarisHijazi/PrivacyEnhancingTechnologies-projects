# @author Faris Hijazi


import inspect
import os
import socket
import sys
import types

from encryption_utils import _bytes_to_string, _string_to_bytes
from utils import recv_msg, AttrDict, send_msg

DESCRIPTION = ("COE449 Assignment3: 1-out-of-2 oblivious transfer."
               "\nClient side: Receiver of the message (Bob)" +
               "\nFaris Hijazi s201578750 26-11-19" +
               "\n=======================================")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# moving the import path 1 directory up (to import utils)
from client_backend import get_arg_parser, get_user_commands, exec_function


# HOST = '127.0.0.1'  # The server's hostname or IP address
# PORT = 65432        # The port used by the server


def main():
    """
    Step 1: Alice generates an RSA key pair PK=(n,e) and SK=(d) and  generates two random  values, r_0 and r_1, and sends them to Bob along with her PK
    Step 2: Bob picks a bit b to be either 0 or 1, and selects r_b
    Step 3: Bob generates a random value k and blinds  r_b by computing  〖v=r〗_b+k^e  mod n and send it to Alice
    Step 4: Alice doesn't know which of  r_0 and r_1 Bob chose.
        She applies them both and come up with two possible values for k:k_0=(v−x_0 )^d  mod n and k_1=(v−x_1 )^d  mod n  Eventually,
        one of these will be equal to  k and can be correctly decrypted by Bob (but not Alice),
        while the other will produce a meaningless random value that does not  reveal any  information about  k
    Step 5: Alice combines the two secret messages with each of the possible keys, m_0^′=m_0+k_0 and m_1^′=m_1+k_1, and sends them both to Bob
    Step 6: Bob knows which of the two messages can be unblinded with  k, so he is able to compute  exactly one of the messages m_b=m_b^′−k
    :return:
    """
    parser = get_arg_parser()
    print(DESCRIPTION + "\n")

    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('connecting to server...', end='')
        s.connect((args.host, args.port))  # connect
        print('\rConnection established ')

        import rsa, secrets, json

        # recv public key
        resp = recv_msg(s)
        resp = json.loads(_bytes_to_string(resp))

        alice_pubkey = rsa.PublicKey(resp['n'], resp['e'])
        r_b_choices = list(map(_string_to_bytes, resp['r_b_choices']))

        print('alice_pubkey', alice_pubkey)
        print('r_b_choices', r_b_choices)

        #    Step 2: Bob picks a bit b to be either 0 or 1, and selects r_bs
        r_b = int.from_bytes(r_b_choices[args.msg_index], 'big')

        #    Step 3: Bob generates a random value k and blinds  r_b by computing  〖v=r〗_b+k^e  mod n and send it to Alice
        k = int.from_bytes(secrets.token_bytes(4), 'big')
        print('k=', k)
        print('r_b=', r_b)
        v = _string_to_bytes(str(alice_pubkey.blind(r_b, k)))

        print('v=', v)
        send_msg(s, v)

        #    Step 6: Bob knows which of the two messages can be unblinded with  k, so he is able to compute  exactly one of the messages m_b=m_b^′−k
        resp6 = recv_msg(s)
        resp6_str = _bytes_to_string(resp6)
        print('resp6_str=', resp6_str)
        combined_list = json.loads(resp6_str)
        combined_list = list(map(_string_to_bytes, combined_list))

        value = combined_list[args.msg_index] - k

        print('the value is:', value)

        if resp in [202]:
            send_msg(s, b'200')  # send OK code
            print('\nTransaction complete')


if __name__ == "__main__":
    main()
