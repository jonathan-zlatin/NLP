import unittest
import main


class TestSolution(unittest.TestCase):
    def test_lemmatize_text(self):
        input_text = ["I have a house in"]
        expected_output = ['I', 'have', 'a', 'house', 'in']
        self.assertEqual(main.lemmatize_text(input_text), expected_output)

    def test_train_unigram_model(self):
        unigram_model = main.train_unigram_model(log_it=False)
        self.assertIsInstance(unigram_model, dict)
        self.assertGreater(len(unigram_model), 0)

    def test_train_bigram_model(self):
        unigram_model = main.train_unigram_model(log_it=False)
        bigram_model = main.train_bigram_model(log_it=False, model_unigram=unigram_model)
        self.assertIsInstance(bigram_model, dict)
        self.assertGreater(len(bigram_model), 0)

    def test_bigram_model_predictions(self):
        # Load trained bigram model
        unigram_model = main.train_unigram_model(log_it=False)
        bigram_model = main.train_bigram_model(log_it=False, model_unigram=unigram_model)

        # Test predictions for next words
        sentence = ["I", "have", "a"]
        next_word = main.bigram_model_predict_next_word(bigram_model, sentence)
        self.assertIsNotNone(next_word)
        self.assertIsInstance(next_word, str)

    def test_linear_interpolation_model(self):
        # Load trained unigram and bigram models
        unigram_model = main.train_unigram_model(log_it=False)
        bigram_model = main.train_bigram_model(log_it=False, model_unigram=unigram_model)

        # Test linear interpolation model
        sentence = ["I", "have", "a", "house"]
        log_probability = main.liner_interpolation_model(unigram_model, bigram_model, sentence, log_it=True)
        self.assertIsInstance(log_probability, float)

        # Test with different lambda values
        custom_lambda_bigram = 0.5
        custom_lambda_unigram = 0.5
        log_probability_custom_lambda = main.liner_interpolation_model(
            unigram_model, bigram_model, sentence, log_it=True, lambda_bigram=custom_lambda_bigram,
            lambda_unigram=custom_lambda_unigram
        )
        self.assertIsInstance(log_probability_custom_lambda, float)


if __name__ == '__main__':
    unittest.main()
