from setuptools import setup

setup(name='tsn_scheduler',  # secondary directory
      version='0.1',
      author='Zheyu Liu',
      author_email='lzyhululu@163.com',
      description='Schelduler using multi_agent reinforcement agent, regard TT flow as AGENT',
      install_requires=['gym',
                        'numpy',
                        'pygame',
                        'pandas',
                        'tensorflow',
                        'matplotlib']  # dependencies
      )
