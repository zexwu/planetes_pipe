"""keywords.py: card names, indexed entries, comments that fit, category order."""
import unittest
import warnings

from astropy.io import fits

from planetes_pipe.flat import QC_FLUX
from planetes_pipe.keywords import DIT, INSTRUME, SHUTTER, Keyword, sorted_by_category


class KeywordTests(unittest.TestCase):
    def test_short_names_map_to_fits_cards(self):
        self.assertEqual(DIT.card, "HIERARCH ESO DET WIN1 DIT1")
        self.assertEqual(INSTRUME.card, "INSTRUME")
        self.assertEqual(SHUTTER.at(4).card, "HIERARCH ESO INS SHUT4 ST")
        self.assertEqual(SHUTTER.at(4).comment, "Shutter T4 open")

    def test_indexed_entries_fill_name_and_comment_positionally(self):
        keyword = QC_FLUX.at(1, "4")  # First QC flux is telescope T4.
        self.assertEqual(keyword.name, "QC.FLATFLUX.SC1")
        self.assertEqual(keyword.comment, "Total flux per DIT, T4")

    def test_set_writes_unit_and_fits_the_comment_in_one_card(self):
        header = fits.Header()
        DIT.set(header, 1.5)
        self.assertEqual(header.comments[DIT.card], "[s] Integration time")
        self.assertEqual(DIT.get(header), 1.5)
        long = Keyword("QC.P2VM_COHERENCE_SC43", "x" * 100)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            long.set(header, 0.9443228394832154)
            self.assertLessEqual(len(header.cards[long.card].image), 80)

    def test_cards_are_ordered_by_category(self):
        header = fits.Header()
        Keyword("QC.X", "QC").set(header, 1)
        DIT.set(header, 1.0)
        SHUTTER.at(1).set(header, True)
        INSTRUME.set(header, "PLANETES")
        order = [card.keyword for card in sorted_by_category(header).cards]
        self.assertEqual(order, ["INSTRUME", "ESO INS SHUT1 ST", "ESO DET WIN1 DIT1", "ESO QC X"])


if __name__ == "__main__":
    unittest.main()
