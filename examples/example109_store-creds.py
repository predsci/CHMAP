
# This is a little bit clunky, but is a better solution than writing passwords into

import os
from cryptography.fernet import Fernet
from settings.app import App

key_file = os.path.join(App.APP_HOME, "settings", "e_key.bin")

# Generate a new local encryption key
# key = Fernet.generate_key()
# print(key)
#
# with open(key_file, 'wb') as file_object:
#     file_object.write(key)

with open(key_file, 'rb') as file_object:
    for line in file_object:
        chd_key = line

# User inputs password interactively so it is never saved
passw = input("Enter a password to encrypt and save: ")

cipher_suite = Fernet(chd_key)
ciphered_text = cipher_suite.encrypt(passw.encode())   #required to be bytes

creds_file = os.path.join(App.APP_HOME, "settings", "e_cred.bin")

with open(creds_file, 'wb') as file_object:
    file_object.write(ciphered_text)



