import keras
import numpy as np
import os
import shutil
import scipy.misc
from keras.models import Model
from data import Load_train_data, Load_validation_data
from experiments import Update_I, PSNR
from net import My_net

def main(
        data_class,
        image_height,
        image_width,
        image_channel,
        batch_size,
        num_iters,
        eval_step,
        validation_path,
        tensorboard,
        unsupervised_loss_weight=None,
        supervised_loss_weight=None,
        unsupervised_number=None,
        supervised_number=None
):
    #load_validation_data and init some variables
    validation_input, validation_ground_truth = Load_validation_data()
    unsupervised_data = None
    supervised_data = None
    ground_truth_data = None
    I = None
    I0 = None
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    # load model
    net = My_net(
        data_class=data_class,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        image_channel=image_channel,
        unsupervised_loss_weight=unsupervised_loss_weight,
        supervised_loss_weight=supervised_loss_weight
    )

    net.network()
    model = net.model

    # train
    best_PSNR = 0
    for iter in range(num_iters):
        if data_class == 'unsupervised':
            if iter % eval_step == 0:
                unsupervised_data = Load_train_data(data_class, unsupervised_number)
                I, I0 = Update_I(unsupervised_data)
                # validate
                predict_model = Model(inputs=model.input, outputs=model.get_layer('add_1').output)
                avg_PSNR = 0
                print('iter: ' + str(iter))
                for i in range(validation_input.shape[0]):
                    validation_image = validation_input[i, :, :, :].reshape((1, image_height, image_width, image_channel))
                    validation_outcome = predict_model.predict({
                        'unsupervised': validation_image,
                        'I': np.zeros((batch_size, image_height, image_width, image_channel)),
                        'I0': np.zeros((batch_size, image_height, image_width, image_channel))
                    })
                    comparision = np.hstack((validation_outcome[0, :, :, 0], validation_ground_truth[i, :, :, 0]))
                    scipy.misc.imsave(validation_path + '/' + str(i) + '.png', comparision)
                    psnr = PSNR(
                        validation_ground_truth[i, :, :, 0].reshape((1, image_height, image_width, image_channel)),
                        validation_outcome,
                        1
                    )
                    print('PSNR: ' + str(psnr))
                    avg_PSNR += psnr
                avg_PSNR /= validation_ground_truth.shape[0]
                with open('validation/validation_loss.txt', 'a+') as f:
                    f.write('iter = ' + str(iter) + '| average_PSNR = ' + str(avg_PSNR) + '\n')
                # save model
                predict_model.save('model/unsupervised_ct_denoise.h5')
                if avg_PSNR > best_PSNR:
                    predict_model.save('model/best_unsupervised_ct_denoise.h5')
                    best_PSNR = avg_PSNR

            model.fit(
                {
                    'unsupervised': unsupervised_data,
                    'I': I,
                    'I0': I0
                },
                np.zeros((unsupervised_number, 1)),
                batch_size=batch_size,
                epochs=1,
                callbacks=[tensorboard]
            )

        elif data_class == 'semisupervised':
            if iter % eval_step == 0:
                unsupervised_data, supervised_data, ground_truth_data = \
                    Load_train_data(data_class, unsupervised_number, supervised_number)
                I, I0 = Update_I(unsupervised_data)
                # validate
                predict_model = Model(inputs=model.input, outputs=model.get_layer('add_1').output)
                avg_PSNR = 0
                print('iter: ' + str(iter))
                for i in range(validation_input.shape[0]):
                    validation_image = validation_input[i, :, :, :].reshape((1, image_height, image_width, image_channel))
                    validation_outcome = predict_model.predict({
                        'unsupervised': validation_image,
                        'I': np.zeros((batch_size, image_height, image_width, image_channel)),
                        'I0': np.zeros((batch_size, image_height, image_width, image_channel)),
                        'supervised': np.zeros((batch_size, image_height, image_width, image_channel)),
                        'ground_truth': np.zeros((batch_size, image_height, image_width, image_channel))
                    })
                    comparision = np.hstack((validation_outcome[0, :, :, 0], validation_ground_truth[i, :, :, 0]))
                    scipy.misc.imsave(validation_path + '/' + str(i) + '.png', comparision)
                    psnr = PSNR(
                        validation_ground_truth[i, :, :, 0].reshape((1, image_height, image_width, image_channel)),
                        validation_outcome,
                        1
                    )
                    print('PSNR: ' + str(psnr))
                    avg_PSNR += psnr
                avg_PSNR /= validation_ground_truth.shape[0]
                with open('validation/validation_loss.txt', 'a+') as f:
                    f.write('iter = ' + str(iter) + '| average_PSNR = ' + str(avg_PSNR) + '\n')
                # save model
                predict_model.save('model/semisupervised_ct_denoise.h5')
                if avg_PSNR > best_PSNR:
                    predict_model.save('model/best_semisupervised_ct_denoise.h5')
                    best_PSNR = avg_PSNR

            model.fit(
                {
                    'supervised': supervised_data,
                    'unsupervised': unsupervised_data,
                    'I': I,
                    'I0': I0,
                    'ground_truth': ground_truth_data
                },
                [np.zeros((unsupervised_number, 1)), np.zeros((unsupervised_number, 1))],
                batch_size=batch_size,
                epochs=1,
                callbacks=[tensorboard]
            )

        else:
            if iter % eval_step == 0:
                supervised_data, ground_truth_data = Load_train_data(data_class, None, supervised_number)
                # validate
                predict_model = Model(inputs=model.input, outputs=model.get_layer('add_1').output)
                avg_PSNR = 0
                print('iter: ' + str(iter))
                for i in range(validation_input.shape[0]):
                    validation_image = validation_input[i, :, :, :].reshape((1, image_height, image_width, image_channel))
                    validation_outcome = predict_model.predict({
                        'supervised': validation_image,
                        'ground_truth': np.zeros((batch_size, image_height, image_width, image_channel))
                    })
                    comparision = np.hstack((validation_outcome[0, :, :, 0], validation_ground_truth[i, :, :, 0]))
                    scipy.misc.imsave(validation_path + '/' + str(i) + '.png', comparision)
                    psnr = PSNR(
                        validation_ground_truth[i, :, :, 0].reshape((1, image_height, image_width, image_channel)),
                        validation_outcome,
                        1
                    )
                    print('PSNR: ' + str(psnr))
                    avg_PSNR += psnr
                avg_PSNR /= validation_ground_truth.shape[0]
                with open('validation/validation_loss.txt', 'a+') as f:
                    f.write('iter = ' + str(iter) + '| average_PSNR = ' + str(avg_PSNR) + '\n')
                # save model
                predict_model.save('model/semisupervised_ct_denoise.h5')
                if avg_PSNR > best_PSNR:
                    predict_model.save('model/best_supervised_ct_denoise.h5')
                    best_PSNR = avg_PSNR

            model.fit(
                {
                    'supervised': supervised_data,
                    'ground_truth': ground_truth_data
                },
                np.zeros((supervised_number, 1)),
                batch_size=batch_size,
                epochs=1,
                callbacks=[tensorboard]
            )

if os.path.exists('model/logs'):
    shutil.rmtree('model/logs')
    os.makedirs('model/logs')
if os.path.exists('validation/validation_loss.txt'):
    with open('validation/validation_loss.txt', 'r+') as f:
        f.truncate()
main(
    data_class='supervised',
    image_height=736,
    image_width=1152,
    image_channel=1,
    batch_size=1,
    num_iters=100000,
    eval_step=5,
    validation_path='validation',
    tensorboard=keras.callbacks.TensorBoard(
            log_dir='model/logs',
            histogram_freq=0,
            write_graph=False,
            write_images=False),
    supervised_loss_weight=1,
    supervised_number=20,

)











            #




