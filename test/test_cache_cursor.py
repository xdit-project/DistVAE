"""Where the causal decoders are up to in their feature cache, and who owns that position"""

import unittest

from distvae.utils import cache_cursor


class TestCacheCursor(unittest.TestCase):

    def test_omitting_a_cursor_gets_a_fresh_one_every_time(self):
        first, second = cache_cursor(None), cache_cursor(None)
        self.assertEqual(first, [0])
        self.assertEqual(second, [0])
        # Identity matters because blocks advance the cursor in place while walking the cache.
        # Sharing one list would make the second decode start where the first stopped.
        self.assertIsNot(first, second)

    def test_a_cursor_handed_in_is_the_one_used(self):
        # A caller walking the cache itself passes its own position in, and gets it back to keep
        # advancing rather than a copy that strands its progress here.
        mine = [7]
        self.assertIs(cache_cursor(mine), mine)


if __name__ == "__main__":
    unittest.main()
