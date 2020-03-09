# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime
from analysis.compare_gt import align_gt
from reconstruction import synchronization as sync


# Load trajectories
data_file = ''
with open(data_file, 'rb') as file:
    flight = pickle.load(file)

# Analysis


print('Finish!')
