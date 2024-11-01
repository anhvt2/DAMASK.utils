
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 1, 3, offset="(0,0,0)", to="(0,0,0)", height=1.5, depth=1, width=1),
    
    # to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=1.5, depth=1, width=1, caption="ReLU"),
    to_Conv("conv2", 1, 16, offset="(2,0,0)", to="(0,0,0)", height=8, depth=1, width=1),
    
    to_connection( "conv1", "conv2"), 
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=8, depth=1, width=1, caption="ReLU"),

    to_Conv("conv3", 1, 64, offset="(4,0,0)", to="(0,0,0)", height=32, depth=1, width=1),
    to_connection( "pool2", "conv3"), 
    to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)",  height=32, depth=1, width=1, caption="ReLU"),

    to_Conv("conv4", 1, 128, offset="(6,0,0)", to="(0,0,0)", height=64, depth=1, width=1),
    to_connection( "pool3", "conv4"), 
    to_Pool("pool4", offset="(0,0,0)", to="(conv4-east)",  height=64, depth=1, width=1, caption="ReLU"),

    to_Conv("conv5", 1, 300, offset="(8,0,0)", to="(0,0,0)", height=96, depth=1, width=1),
    to_connection( "pool4", "conv5"), 


    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
