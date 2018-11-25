disable_embedding = False
TEST_EMBEDDING = '''
pass
''' if disable_embedding else '''
import IPython
IPython.embed()
assert(0)
'''