#!/usr/bin/env python

import os
import full_line_graph_test

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    full_line_graph_test.run()
