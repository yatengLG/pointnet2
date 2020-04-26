# -*- coding: utf-8 -*-
# @Author  : LG

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name = 'LG',
    version = '0.0.1',
    ext_modules = [CppExtension('LG',sources=['LG_module.cpp'])],
    cmdclass = {'build_ext': BuildExtension}
)
