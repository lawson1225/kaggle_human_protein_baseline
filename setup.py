from setuptools import setup

setup(
   name='kaggle_human_protein_baseline',
   version='1.0',
   description='bninception',
   author='LawsonChen',
   author_email='lawson901225@gmail.com',
   packages=['kaggle_human_protein_baseline'],  #same as name
   install_requires=['imgaug'], #external packages as dependencies
   # scripts=[
   #          'scripts/cool',
   #          'scripts/skype',
   #         ]
)