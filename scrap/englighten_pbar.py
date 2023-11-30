import random
import time
import enlighten

bar_format = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
             u'S:{count_0:{len_total}d} ' + \
             u'E:{count_1:{len_total}d} ' + \
             u'F:{count_2:{len_total}d} ' + \
             u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

manager = enlighten.get_manager()
success = manager.counter(total=100, desc='Testing', unit='tests',
                            color='green', bar_format=bar_format)
errors = success.add_subcounter('white')
failures = success.add_subcounter('red')

while success.count < 100:
    time.sleep(random.uniform(0.1, 0.3))  # Random processing time
    result = random.randint(0, 10)

    if result == 7:
        errors.update()
    if result in (5, 6):
        failures.update()
    else:
        success.update()