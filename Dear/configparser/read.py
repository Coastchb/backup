#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:28:30 2017

@author: dear
"""

import configparser
config = configparser.ConfigParser()

print(config.sections())

config.read('example.ini')

print(config.sections())

print('bitbucket.org' in config)

print('bytebong.com' in config)

print(config['bitbucket.org']['User'])

print(config['DEFAULT']['Compression'])

topsecret = config['topsecret.server.com']
print(topsecret['ForwardX11'])

print(topsecret['Port'])

for key in config['bitbucket.org']: print(key)

print(config['bitbucket.org']['ForwardX11'])

