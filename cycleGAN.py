gen_iterations = 0
epoch = 0
database_name='';
load_pretraining_weights = False
import os
os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'
#os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_compile,dnn.library_path=/usr/lib'
#============================================================================
import keras.backend as K
if os.environ['KERAS_BACKEND'] =='theano':
    channel_axis=1
    K.set_image_data_format('channels_first')
    channel_first = True
else:
    K.set_image_data_format('channels_last')
    channel_axis=-1
    channel_first = False

from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
#============================================================================
# Weights initializations
# bias are initailized as 0
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization
#============================================================================
# HACK speed up theano
if K._BACKEND == 'theano':
    import keras.backend.theano_backend as theano_backend
    def _preprocess_conv2d_kernel(kernel, data_format):
        #return kernel
        if hasattr(kernel, "original"):
            print("use original")
            return kernel.original
        elif hasattr(kernel, '_keras_shape'):
            s = kernel._keras_shape
            print("use reshape",s)
            kernel = kernel.reshape((s[3], s[2],s[0], s[1]))
        else:
            kernel = kernel.dimshuffle((3, 2, 0, 1))
        return kernel
    theano_backend._preprocess_conv2d_kernel = _preprocess_conv2d_kernel
#============================================================================
# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    
    if channel_first:
        input_a =  Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in))
    '''
    # The layers in the model are connected pairwise.
    # A bracket notation is used, such that 
    # after the layer is created, the layer from which the input to 
    # the current layer comes from is specified.
    # example
    # visible = Input(shape=(2,))
    # hidden = Dense(2)(visible) 
    '''
    _ = input_a
    # kernel_size=4 <=> kernel_size=(4,4)
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    '''
    # LeakyReLU
    # f(x) = alpha * x for x < 0
    # f(x) = x for x >= 0
    '''
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)#??
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid" if use_sigmoid else None) (_)    
    return Model(inputs=[input_a], outputs=_)
#============================================================================
def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):    
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        '''
        # assert True # nothing happens
        # assert False => AssertionError
        '''
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])            
        x = Activation("relu")(x)
        '''
        # Transposed convolution layer (sometimes called Deconvolution).
        # keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), 
        #                              padding='valid', data_format=None, 
        #                              activation=None, use_bias=True, 
        #                              kernel_initializer='glorot_uniform',
        #                              bias_initializer='zeros', kernel_regularizer=None, 
        #                              bias_regularizer=None, activity_regularizer=None,
        #                              kernel_constraint=None, bias_constraint=None)
        '''
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)        
        '''
        # It crops along spatial dimensions, i.e. width and height.
        # If int: the same symmetric cropping is applied to width and height.
        # keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
        '''
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        # x = a type of layer
        return x
    
    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))        
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])
#============================================================================
nc_in = 3
nc_out = 3
ngf = 64
ndf = 64
use_lsgan = True
λ = 10 if use_lsgan else 100

loadSize = 143
imageSize = 128
batchSize = 1
lrD = 2e-4
lrG = 2e-4
#============================================================================
netDA = BASIC_D(nc_in, ndf, use_sigmoid = not use_lsgan)
netDB = BASIC_D(nc_out, ndf, use_sigmoid = not use_lsgan)
netDA.summary()# prints a summary representation of your model. 
#============================================================================
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


netGB = UNET_G(imageSize, nc_in, nc_out, ngf)
netGA = UNET_G(imageSize, nc_out, nc_in, ngf)
#SVG(model_to_dot(netG, show_shapes=True).create(prog='dot', format='svg'))
netGA.summary()
#============================================================================
from keras.optimizers import RMSprop, SGD, Adam
'''
# loss function 
# lambda param1, param2, ... : expression
# func2 = lambda x,y,z : x+y+z
# func2(1, 2, 3) =>6
'''
if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    '''
    # keras.backend.function(inputs, outputs, updates=None)
    # Instantiates a Keras function.
    # Returns: Output values as Numpy arrays.
    '''
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate

real_A, fake_B, rec_A, cycleA_generate = cycle_variables(netGB, netGA)
real_B, fake_A, rec_B, cycleB_generate = cycle_variables(netGA, netGB)
#============================================================================
# see literature page 5
def D_loss(netD, real, fake, rec):
    # output_real: Discriminator output=1
    output_real = netD([real])
    # output_fake: Discriminator output=0
    output_fake = netD([fake])
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))# E_y[(D(y)-1)^2]
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))# E_x[(D(G(x))^2]
    loss_G = loss_fn(output_fake, K.ones_like(output_fake))# E_x[(D(G(x)-1)^2]
    loss_D = loss_D_real+loss_D_fake
    loss_cyc = K.mean(K.abs(rec-real))# cycle loss at literature eq.2
    return loss_D, loss_G, loss_cyc

loss_DA, loss_GA, loss_cycA = D_loss(netDA, real_A, fake_A, rec_A)
loss_DB, loss_GB, loss_cycB = D_loss(netDB, real_B, fake_B, rec_B)
loss_cyc = loss_cycA+loss_cycB
#============================================================================
loss_G = loss_GA+loss_GB+λ*loss_cyc
loss_D = loss_DA+loss_DB

weightsD = netDA.trainable_weights + netDB.trainable_weights
weightsG = netGA.trainable_weights + netGB.trainable_weights

training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD,[],loss_D)
netD_train = K.function([real_A, real_B],[loss_DA/2, loss_DB/2], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsG,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], training_updates)
#============================================================================
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle

def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn):
    im = Image.open(fn).convert('RGB')
    im = im.resize( (loadSize, loadSize), Image.BILINEAR )
    arr = np.array(im)/255*2-1
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2
    img = arr[h1:h2, w1:w2, :]
    if randint(0,1):
        img=img[:,::-1]
    if channel_first:        
        img = np.moveaxis(img, 2, 0)
    return img

#data = "edges2shoes"
'''
data = "horse2zebra"
train_A = load_data('{}/trainA/*.jpg'.format(data))
train_B = load_data('{}/trainB/*.jpg'.format(data))
'''
'''
data = "ISTD_Dataset"
train_A = load_data('../shadow_Dataset/{}/train/train_A/*.png'.format(data))
train_B = load_data('../shadow_Dataset/{}/train/train_C/*.png'.format(data))
'''
data = "SRD"
train_A = load_data('../shadow_Dataset/{}/test_data/shadow/*.jpg'.format(data))
train_B = load_data('../shadow_Dataset/{}/test_data/shadow_free/*.jpg'.format(data))
# train_A,train_B is a file name list
assert len(train_A) and len(train_B)
#============================================================================
def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None    
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            # random data list
            shuffle(data)
            i = 0
            #mymymymymymymymymymymymymymymymymymy
            if(((epoch%8)==0)or(epoch==(niter-1))):
                if not os.path.exists('CGAN_{}_weight/{}'.format(database_name,epoch)):
                    os.makedirs('CGAN_{}_weight/{}'.format(database_name,epoch))
                netDA.save_weights('CGAN_{}_weight/{}/netDA_epoch{}.h5'.format(database_name,epoch,epoch))
                netDB.save_weights('CGAN_{}_weight/{}/netDB_epoch{}.h5'.format(database_name,epoch,epoch))
                netGA.save_weights('CGAN_{}_weight/{}/netGA_epoch{}.h5'.format(database_name,epoch,epoch))
                netGB.save_weights('CGAN_{}_weight/{}/netGB_epoch{}.h5'.format(database_name,epoch,epoch))
            #mymymymymymymymymymymymymymymymymymy
            epoch+=1        
        rtn = [read_image(data[j]) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)       

def minibatchAB(dataA, dataB, batchsize):
    batchA=minibatch(dataA, batchsize)
    batchB=minibatch(dataB, batchsize)
    tmpsize = None    
    while True:        
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B
#============================================================================
from IPython.display import display
def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    result = Image.fromarray(int_X)
    #mymymymymymymymymymymymymymymymymymymymymymymymymymymymy
    if(((epoch%8)==0)or(epoch==(niter-1))):
        if not os.path.exists('CGAN_{}_img/{}'.format(database_name,epoch)):
            os.makedirs('CGAN_{}_img/{}'.format(database_name,epoch))
        result.save('CGAN_{}_img/{}/output{}.jpg'.format(database_name,epoch,gen_iterations))
    #mymymymymymymymymymymymymymymymymymymymymymymymymymymymymymy
    display(result)
    #display(Image.fromarray(int_X))
#============================================================================
train_batch = minibatchAB(train_A, train_B, 6)

_, A, B = next(train_batch)
showX(A)
showX(B)
del train_batch, A, B
#============================================================================
def showG(A,B):
    assert A.shape==B.shape
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]        
    rA = G(cycleA_generate, A)
    rB = G(cycleB_generate, B)
    arr = np.concatenate([A,B,rA[0],rB[0],rA[1],rB[1]])

    showX(arr, 3)
#============================================================================
import time
from datetime import datetime
from IPython.display import clear_output
t0 = time.time()
niter = 150
#gen_iterations = 0
#epoch = 0
errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
display_iters = 50
#val_batch = minibatch(valAB, 6, direction)
train_batch = minibatchAB(train_A, train_B, batchSize)
#mymymymymymymymymymymymymymymymymymymymymymymymymymymymymymy
if load_pretraining_weights:
    netGA.load_weights('CGAN_shadow_weight/149/netGA_epoch149.h5')
    netGB.load_weights('CGAN_shadow_weight/149/netGB_epoch149.h5')
    netDA.load_weights('CGAN_shadow_weight/149/netDA_epoch149.h5')
    netDB.load_weights('CGAN_shadow_weight/149/netDB_epoch149.h5')
#mymymymymymymymymymymymymymymymymymymymymymymymymymymymymymy
while epoch < niter: 
    epoch, A, B = next(train_batch)        
    errDA, errDB  = netD_train([A, B])
    errDA_sum +=errDA
    errDB_sum +=errDB
    aaaa=netDA(A)
    # epoch, trainA, trainB = next(train_batch)
    errGA, errGB, errCyc = netG_train([A, B])
    errGA_sum += errGA
    errGB_sum += errGB
    errCyc_sum += errCyc
    gen_iterations+=1
    if gen_iterations%display_iters==0:
        #if gen_iterations%(5*display_iters)==0:
        clear_output()
        print('[%d/%d][%d] Loss_D: %f %f Loss_G: %f %f loss_cyc %f'
        % (epoch, niter, gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, 
           errCyc_sum/display_iters), time.time()-t0)
        
        #mymymymymymymymymymymymymymymymymymymymymymymymymymymymymymy
        if not os.path.exists('CGAN_{}_weight/error_log.txt'.format(database_name)):
            ppend_write = 'w' # make a new file if not
        else:
            append_write = 'a' # append if already exists
        with open("error_log.txt", append_write") as myfile:
            myfile.write("%f\t%f\t%f\t%f\t%f\t"
                         % ( errDA_sum/display_iters, errDB_sum/display_iters,
                            errGA_sum/display_iters, errGB_sum/display_iters, 
                            errCyc_sum/display_iters))
            myfile.write("{}\n".format(str(datetime.now())))
        #mymymymymymymymymymymymymymymymymymymymymymymymymymymymymymy
        
        _, A, B = train_batch.send(4)
        showG(A,B)        
        errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
