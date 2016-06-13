import unittest

class TestModules(unittest.TestCase):
    def test_variance_decomposition_import(self):
        try:
            import limix.modules.varianceDecomposition as vd
        except ImportError:
            self.fail()

if __name__ == '__main__':
    unittest.main()
