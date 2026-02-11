import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    'font.size': 16,  # Global font size for text
})

# Data extracted from the table in the image
cases = [
    "Fully Trusted Data Trust Assessment Function",
    "Fully Un-Trusted Data Trust Assessment Function",
    "Fully Dis-Trusted Data Trust Assessment Function",
]


operation = [(0.856, 0.042, 0.102),
             (0.287, 0.05, 0.663),
             (0.0, 1.0, 0.0),
             ]

FFAgrtrusted = [(0.846, 0.053, 0.102),
                (0.298, 0.043, 0.659),
                (0.0, 1.0, 0.0),
                ]
 
FFAgruntrusted = [(0.839, 0.053, 0.108),
                  (0.296, 0.043, 0.661),
                  (0.0, 1.0, 0.0)
                  ]

FFAgrdistrusted = [(0.0, 1.0, 0.0),
                   (0.0, 1.0, 0.0),
                   (0.0, 1.0, 0.0)]

# Plotting histograms for each case
for i, case in enumerate(cases):
    labels = ["Belief", "Disbelief", "Uncertainty"]
    fully_trusted_values = FFAgrtrusted[i]
    fully_untrusted_values = FFAgruntrusted[i]
    fully_distrusted_values  = FFAgrdistrusted[i]
    
    operationval = operation[i]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x - 2*width, operationval, width, label='Operation on the first data of the test dataset')
    ax.bar(x - width, fully_trusted_values, width, label='Feed Forward + Aggregation on Tx fully trust opinion')
    ax.bar(x , fully_untrusted_values, width, label='Feed Forward + Aggregation on Tx fully un-trust opinion')
    ax.bar(x + width , fully_distrusted_values, width, label='Feed Forward + Aggregation on Tx fully dis-trust opinion')


    ax.set_ylabel('Values')
    ax.set_title(f'Histogram of {case}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()
