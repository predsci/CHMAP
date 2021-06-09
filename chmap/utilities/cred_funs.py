
# credential handling functions

import os
from cryptography.fernet import Fernet


def recover_pwd(cred_dir):
    if cred_dir is None:
        proj_dir = os.path.dirname(os.path.abspath(__file__))
        cred_dir = os.path.join(proj_dir, '..', 'settings')

    key_file = os.path.join(cred_dir, "e_key.bin")

    with open(key_file, 'rb') as file_object:
        for line in file_object:
            chd_key = line

    cipher_suite = Fernet(chd_key)

    creds_file = os.path.join(cred_dir, "e_cred.bin")

    with open(creds_file, 'rb') as file_object:
        for line in file_object:
            ciphered_text = line

    unciphered_text = (cipher_suite.decrypt(ciphered_text))

    plain_text_password = bytes(unciphered_text).decode("utf-8")

    return plain_text_password

