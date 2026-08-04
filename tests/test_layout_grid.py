import unittest

from medical_kg_sourceprep.layout_grid import LayoutBlock, table_grids


class LayoutGridTests(unittest.TestCase):
    def test_html_rowspan_and_colspan_expand_to_rectangular_grid(self) -> None:
        block = LayoutBlock(
            0,
            4,
            "table",
            """<table><tr><th rowspan="2">项目</th><th colspan="2">结果信息</th></tr>
            <tr><th>结果</th><th>单位</th></tr><tr><td>ALT</td><td>8</td><td>U/L</td></tr></table>""",
            (0, 0, 100, 100),
        )
        grid = table_grids((block,))[0]
        self.assertEqual(len(grid.rows), 3)
        self.assertEqual({len(row) for row in grid.rows}, {3})
        self.assertEqual(grid.rows[0][0].text, "项目")
        self.assertEqual(grid.rows[1][0].text, "项目")
        self.assertEqual(grid.rows[0][1].source_ref, grid.rows[0][2].source_ref)

    def test_pipe_markdown_normalizes_to_same_grid_contract(self) -> None:
        block = LayoutBlock(
            2,
            3,
            "table",
            "| 项目 | 结果 | 单位 |\n| --- | ---: | --- |\n| ALT | 8 | U/L |",
            None,
        )
        grid = table_grids((block,))[0]
        self.assertEqual([[cell.text for cell in row] for row in grid.rows], [
            ["项目", "结果", "单位"],
            ["ALT", "8", "U/L"],
        ])
        self.assertEqual(grid.rows[1][1].source_ref, "p2.b3.r1.c1")


if __name__ == "__main__":
    unittest.main()
