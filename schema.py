import numpy as np

player_dt = np.dtype([('id', 'int16'),  
                      ('position', 'f4', (2,)), 
                      ('velocity', 'f4', (2,)), 
                      ('team', 'U8'), 
                      ('i_p', 'f4')])