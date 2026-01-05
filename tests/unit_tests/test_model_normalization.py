import unittest
from upsonic.models import normalize_model_id, _build_model_alias_index, _get_cached_known_providers


class TestNormalizeModelId(unittest.TestCase):

    def test_bedrock_claude_3_5_sonnet_with_version(self):
        result = normalize_model_id('bedrock/claude-3-5-sonnet:v2')
        self.assertEqual(result, 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')

    def test_bedrock_claude_3_5_sonnet_default_latest(self):
        result = normalize_model_id('bedrock/claude-3-5-sonnet')
        self.assertEqual(result, 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')

    def test_bedrock_claude_3_opus_with_version(self):
        result = normalize_model_id('bedrock/claude-3-opus:v1')
        self.assertEqual(result, 'bedrock/anthropic.claude-3-opus-20240229-v1:0')

    def test_bedrock_claude_3_opus_default_latest(self):
        result = normalize_model_id('bedrock/claude-3-opus')
        self.assertEqual(result, 'bedrock/anthropic.claude-3-opus-20240229-v1:0')

    def test_bedrock_claude_3_haiku_with_version(self):
        result = normalize_model_id('bedrock/claude-3-haiku:v1')
        self.assertEqual(result, 'bedrock/anthropic.claude-3-haiku-20240307-v1:0')

    def test_bedrock_claude_3_5_haiku_with_version(self):
        result = normalize_model_id('bedrock/claude-3-5-haiku:v1')
        self.assertEqual(result, 'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0')

    def test_bedrock_prefers_non_prefixed_models(self):
        result = normalize_model_id('bedrock/claude-3-5-sonnet:v2')
        self.assertNotIn('us.', result)
        self.assertNotIn('eu.', result)
        self.assertNotIn('global.', result)

    def test_openai_gpt4o_with_latest(self):
        result = normalize_model_id('openai/gpt-4o:latest')
        self.assertEqual(result, 'openai/gpt-4o')

    def test_openai_gpt4o_default_latest(self):
        result = normalize_model_id('openai/gpt-4o')
        self.assertEqual(result, 'openai/gpt-4o')

    def test_anthropic_claude_already_full_id(self):
        result = normalize_model_id('anthropic/claude-3-5-haiku-latest')
        self.assertEqual(result, 'anthropic/claude-3-5-haiku-latest')

    def test_ollama_passthrough_unknown_provider(self):
        result = normalize_model_id('ollama/llama3.1:8b')
        self.assertEqual(result, 'ollama/llama3.1:8b')

    def test_ollama_passthrough_with_size(self):
        result = normalize_model_id('ollama/llama3.1:70b')
        self.assertEqual(result, 'ollama/llama3.1:70b')

    def test_ollama_passthrough_qwen(self):
        result = normalize_model_id('ollama/qwen3:30b')
        self.assertEqual(result, 'ollama/qwen3:30b')

    def test_nvidia_passthrough_meta_llama(self):
        result = normalize_model_id('nvidia/meta/llama-3.1-nemotron-70b-instruct:1.0')
        self.assertEqual(result, 'nvidia/meta/llama-3.1-nemotron-70b-instruct:1.0')

    def test_nvidia_passthrough_mistral(self):
        result = normalize_model_id('nvidia/mistral/mistral-large:2.0')
        self.assertEqual(result, 'nvidia/mistral/mistral-large:2.0')

    def test_nvidia_passthrough_openai_model(self):
        result = normalize_model_id('nvidia/openai/gpt-oss-20b:1.0')
        self.assertEqual(result, 'nvidia/openai/gpt-oss-20b:1.0')

    def test_known_model_exact_match_unchanged(self):
        result = normalize_model_id('openai/gpt-4o-2024-05-13')
        self.assertEqual(result, 'openai/gpt-4o-2024-05-13')

    def test_no_slash_passthrough(self):
        result = normalize_model_id('gpt-4o')
        self.assertEqual(result, 'gpt-4o')

    def test_groq_passthrough(self):
        result = normalize_model_id('groq/llama-3.1-8b-instant')
        self.assertEqual(result, 'groq/llama-3.1-8b-instant')

    def test_mistral_provider(self):
        result = normalize_model_id('mistral/mistral-large-latest')
        self.assertEqual(result, 'mistral/mistral-large-latest')


class TestKnownProviders(unittest.TestCase):

    def test_bedrock_is_known_provider(self):
        providers = _get_cached_known_providers()
        self.assertIn('bedrock', providers)

    def test_openai_is_known_provider(self):
        providers = _get_cached_known_providers()
        self.assertIn('openai', providers)

    def test_anthropic_is_known_provider(self):
        providers = _get_cached_known_providers()
        self.assertIn('anthropic', providers)

    def test_ollama_is_not_known_provider(self):
        providers = _get_cached_known_providers()
        self.assertNotIn('ollama', providers)

    def test_nvidia_is_not_known_provider(self):
        providers = _get_cached_known_providers()
        self.assertNotIn('nvidia', providers)


class TestModelAliasIndex(unittest.TestCase):

    def test_alias_index_not_empty(self):
        index = _build_model_alias_index()
        self.assertGreater(len(index), 0)

    def test_bedrock_claude_3_5_sonnet_in_index(self):
        index = _build_model_alias_index()
        self.assertIn('bedrock/claude-3-5-sonnet', index)

    def test_bedrock_claude_3_opus_in_index(self):
        index = _build_model_alias_index()
        self.assertIn('bedrock/claude-3-opus', index)

    def test_bedrock_alias_has_latest_version(self):
        index = _build_model_alias_index()
        self.assertIn('latest', index['bedrock/claude-3-5-sonnet'])

    def test_bedrock_alias_latest_is_non_prefixed(self):
        index = _build_model_alias_index()
        latest = index['bedrock/claude-3-5-sonnet']['latest']
        self.assertFalse(latest.startswith('bedrock/us.'))
        self.assertFalse(latest.startswith('bedrock/eu.'))
        self.assertFalse(latest.startswith('bedrock/global.'))


if __name__ == '__main__':
    unittest.main()
