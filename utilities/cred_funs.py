
# credential handling functions

import os
from cryptography.fernet import Fernet
from chmap.settings.app import App


def recover_pwd():

    key_file = os.path.join(App.APP_HOME, "settings", "e_key.bin")

    with open(key_file, 'rb') as file_object:
        for line in file_object:
            chd_key = line

    cipher_suite = Fernet(chd_key)

    creds_file = os.path.join(App.APP_HOME, "settings", "e_cred.bin")

    with open(creds_file, 'rb') as file_object:
        for line in file_object:
            ciphered_text = line

    unciphered_text = (cipher_suite.decrypt(ciphered_text))

    plain_text_password = bytes(unciphered_text).decode("utf-8")

    return plain_text_password

