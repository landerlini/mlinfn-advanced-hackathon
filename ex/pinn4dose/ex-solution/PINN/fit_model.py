import tensorflow as tf
import numpy as np
#from tensorflow import keras
import os
import datetime

date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name='experiment_'+date

#####################################################################################

#####################################################################################
####################### here define your pde equation system! #######################
# pde_res: partial equation loss
# IC_loss: Initial condition loss
# bc_loss: boundary condition loss
# initial fucntion: function to interpolate the initial condition cs has to be a 
# scipy interpolating function such as scipy.interpolate.CubicSpline
#####################################################################################

def initial_2(x,y,cs):
    z=cs(x,y, grid=False)
    return np.expand_dims(z,-1)

@tf.function
def pde_res(net,x,y,t, alpha,source):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x,y,t])
        u=net(tf.stack([x,y,t],-1), training=True)
        u_x=tape.gradient(u,x)
        u_y=tape.gradient(u,y)
    u_t=tape.gradient(u,t)    
    u_xx=tape.gradient(u_x,x)
    u_yy=tape.gradient(u_y,y)
    
    res_loss=tf.reduce_mean(tf.square(u_t-alpha*(u_xx+u_yy)-source))
    return res_loss

@tf.function
def IC_loss(net,x,y, t, true_ic):
    pred_ic=net(tf.stack([x,y,t],-1), training=True)
    ic_loss=tf.reduce_mean(tf.square(pred_ic-true_ic))
    return ic_loss

@tf.function
def BC_loss(net, x,y,t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x,y])
        u=net(tf.stack([x,y,t],-1), training=True)
    u_x=tape.gradient(u,x)
    u_y=tape.gradient(u,y)
    
    bc_loss=tf.reduce_mean(tf.square(u_x+u_y))
    return bc_loss

def pde_loss(net,point1,point2,point3,cs, alpha=0.005, source=0):
    x=tf.constant(point1[:,0],tf.float32)
    y=tf.constant(point1[:,1],tf.float32)
    t=tf.constant(point1[:,2],tf.float32)
    alpha=tf.constant(alpha, tf.float32)
    source=tf.constant(source, tf.float32)
    res_loss=pde_res(net,x,y,t, alpha,source)

    xic=tf.constant(point2[:,0],tf.float32)
    yic=tf.constant(point2[:,1],tf.float32)
    tic=tf.constant(point2[:,2],tf.float32)
    true_ic=tf.cast(initial_2(point2[:,0],point2[:,1],cs),tf.float32)
    ic_loss=IC_loss(net,xic,yic,tic,true_ic)

    xbc=tf.constant(point3[:,0],tf.float32)
    ybc=tf.constant(point3[:,1],tf.float32)
    tbc=tf.constant(point3[:,2],tf.float32)    
    bc_loss=BC_loss(net,xbc,ybc,tbc)


    total_loss=res_loss+ic_loss+bc_loss
    return total_loss, res_loss, ic_loss, bc_loss

def fit_model(model,geotime,cs,epochs, alpha, source=0,model_name=experiment_name, T : float = 0,n_points=20000, nic_points=20000, nbc_points=1000, optm=tf.keras.optimizers.legacy.Adam()):

    '''
    model      : neural network
    geotime    : geometry class from datagrid.py. It can genarate domain points and boundary points
    cs         : interpolating function for initial condition such as scipy.interpolate.CubicSpline
                 look at interpolating.py
    epochs     : Number of epochs
    alpha      : Diffusion coefficient
    source     : Source term default 0
    model_name : String containing the model name
    T          : Time initial condition (can be different from 0)
    n_points   : number of grid point
    nic_points : number of x point initial condition
    nbc_points : number of boundary point at different time
    optm       : network optmizer algorithm

    '''


    checkpoint_dir = './training_'+model_name
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=optm,
                                    generator=model)
    

    manager=tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    log_dir="logs_"+model_name+"/"
    summary_writer = tf.summary.create_file_writer(
    log_dir + "total_loss/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer_2 = tf.summary.create_file_writer(
    log_dir + 'res_loss/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer_3 = tf.summary.create_file_writer(
    log_dir + 'ic_loss/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer_4 = tf.summary.create_file_writer(
    log_dir + 'bc_loss/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    train_loss_record=[]
    best=np.infty

    for itr in range(epochs):
        point1=geotime.get_random_points(n_points)
        point2=geotime.get_random_points_atsometime(nic_points,T)
        point3=geotime.get_random_boundary_points(nbc_points)


        with tf.GradientTape() as tape:
            total_loss, res_loss, ic_loss, bc_loss=pde_loss(model,point1,point2,point3,cs, alpha, source)
        train_loss_record.append([total_loss.numpy(), res_loss.numpy(), ic_loss.numpy(), bc_loss.numpy()])
        grad_w=tape.gradient(total_loss,model.trainable_variables)
        optm.apply_gradients(zip(grad_w, model.trainable_variables))

        if total_loss<best:
            #print('Find new best! Model checkpoint saving')
            manager.save()

        with summary_writer.as_default():
            tf.summary.scalar('total_loss', total_loss, step=itr+1)
        with summary_writer_2.as_default():
            tf.summary.scalar('total_loss', res_loss, step=itr+1)
        with summary_writer_3.as_default():
            tf.summary.scalar('total_loss', ic_loss, step=itr+1)
        with summary_writer_4.as_default():
            tf.summary.scalar('total_loss', bc_loss, step=itr+1)

        if itr % 100 == 0:
            print(total_loss.numpy(), res_loss.numpy(), ic_loss.numpy(), bc_loss.numpy())
            print('Number of iteration: '+str(itr))
