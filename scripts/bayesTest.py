

from bayes import NaiveBayes
import pandas as pd
import unittest

class TestBayes(unittest.TestCase):
        
    def testExample1(self):
        dataset = pd.read_csv('allelectronics.csv')
        self.model = NaiveBayes(dataset)
        self.model.train()
        datatuple = {'age':'youth','income':'medium','student':'yes','credit_rating':'fair'}
        print(self.model.predict(datatuple))
        self.assertEqual(self.model.predict(datatuple),'yes')
        
if __name__ == '__main__':
    unittest.main()
    

