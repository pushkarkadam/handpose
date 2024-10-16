import torch
import copy 
import sys 

sys.path.append('../')

import handpose 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_box_loss1():
    """Tests box_loss()"""

    x = torch.Tensor([0.5]).reshape(1,1,1,1)
    y = torch.Tensor([0.5]).reshape(1,1,1,1)
    w = torch.Tensor([0.5]).reshape(1,1,1,1)
    h = torch.Tensor([0.5]).reshape(1,1,1,1)

    x_p = copy.deepcopy(x)
    y_p = copy.deepcopy(y)
    w_p = copy.deepcopy(w)
    h_p = copy.deepcopy(h)

    obj_conf = torch.Tensor([1]).reshape(1,1,1,1)

    box_truth = (x, y, w, h)
    box_pred = (x_p, y_p, w_p, h_p)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)

    assert(loss == torch.tensor(0))

def test_box_loss2():
    """Tests box_loss()"""

    x = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)
    y = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)
    w = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)
    h = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)

    x_p = copy.deepcopy(x)
    y_p = copy.deepcopy(y)
    w_p = copy.deepcopy(w)
    h_p = copy.deepcopy(h)

    obj_conf = torch.Tensor([[1,0],[0, 0]]).reshape(1,1,2,2)

    box_truth = (x, y, w, h)
    box_pred = (x_p, y_p, w_p, h_p)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)

    assert(loss == torch.tensor(0))

def test_box_loss3():
    """Tests box_loss()"""

    x = torch.ones((16,1,19,19))
    y = torch.ones((16,1,19,19))
    w = torch.ones((16,1,19,19))
    h = torch.ones((16,1,19,19))

    x_p = copy.deepcopy(x)
    y_p = copy.deepcopy(y)
    w_p = copy.deepcopy(w)
    h_p = copy.deepcopy(h)

    obj_conf = torch.ones((16,1,19,19))

    box_truth = (x, y, w, h)
    box_pred = (x_p, y_p, w_p, h_p)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)

    assert(loss == torch.tensor(0))

def test_box_loss4():
    """Test for box_loss()"""
    x = torch.Tensor([0.5]).reshape(1,1,1,1)
    y = torch.Tensor([0.5]).reshape(1,1,1,1)
    w = torch.Tensor([0.5]).reshape(1,1,1,1)
    h = torch.Tensor([0.5]).reshape(1,1,1,1)

    xp = torch.Tensor([0.4]).reshape(1,1,1,1)
    yp = torch.Tensor([0.4]).reshape(1,1,1,1)
    wp = torch.Tensor([0.4]).reshape(1,1,1,1)
    hp = torch.Tensor([0.4]).reshape(1,1,1,1)

    obj_conf = torch.Tensor([1]).reshape(1,1,1,1)

    box_truth = (x, y, w, h)
    box_pred = (xp, yp, wp, hp)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=1)

    manual_calc = (x - xp)**2 + (y - yp)**2 + (w**(1/2) - wp**(1/2))**2 + (h**(1/2) - hp**(1/2))**2

    torch.testing.assert_close(loss, manual_calc.squeeze())

def test_box_loss5():
    """Test for box_loss()"""
    x = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    y = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    w = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    h = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)

    xp = torch.Tensor([[0.4], [0.3]]).reshape(2,1,1,1)
    yp = torch.Tensor([[0.4], [0.3]]).reshape(2,1,1,1)
    wp = torch.Tensor([[0.4], [0.3]]).reshape(2,1,1,1)
    hp = torch.Tensor([[0.4], [0.3]]).reshape(2,1,1,1)

    obj_conf = torch.Tensor([[1], [1]]).reshape(2,1,1,1)

    box_truth = (x, y, w, h)
    box_pred = (xp, yp, wp, hp)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=1)

    manual_calc = torch.sum((x - xp)**2 + (y - yp)**2 + (w**(1/2) - wp**(1/2))**2 + (h**(1/2) - hp**(1/2))**2) / x.size(0)

    torch.testing.assert_close(loss, manual_calc.squeeze())

def test_conf_loss():
    """Tests conf_loss()"""

    conf_truth = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape((1,1,3,3))
    conf_pred = torch.Tensor([[0.1,0.2,0.4],[0.8,0.9,0.7],[0.2,0.01,0.2]]).reshape((1,1,3,3))

    loss = handpose.loss.conf_loss(conf_truth, conf_pred, 0.5)

    torch.testing.assert_close(loss, torch.tensor(0.7200), rtol=1e-4, atol=1e-4)

def test_class_loss():
    """Tests class_loss()"""

    classes_truth = torch.cat([torch.ones(1,1,3,3), torch.zeros(1,1,3,3)], dim=1)
    classes_pred = torch.cat([torch.ones(1,1,3,3), torch.zeros(1,1,3,3)], dim=1)
    obj_conf = torch.ones((1,1,3,3))

    loss = handpose.loss.class_loss(classes_truth, classes_pred, obj_conf)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_class_loss_mse1():
    """Test class_loss_mse()"""

    classes_truth = torch.cat([torch.ones(1,1,3,3), torch.zeros(1,1,3,3)], dim=1)
    classes_pred = torch.cat([torch.ones(1,1,3,3), torch.zeros(1,1,3,3)], dim=1)
    obj_conf = torch.ones((1,1,3,3))

    loss = handpose.loss.class_loss_mse(classes_truth, classes_pred, obj_conf)

    torch.testing.assert_close(loss, torch.tensor(0.0))

def test_class_loss_mse2():
    """Test class_loss_mse()"""
    
    obj_conf = torch.ones((1,1,3,3))

    classes_truth = torch.cat([torch.zeros(1,1,3,3), torch.ones(1,1,3,3)], dim=1)
    classes_pred = torch.cat([torch.zeros(1,1,3,3), torch.ones(1,1,3,3)], dim=1)
    classes_pred[0,0,1,1] = 1
    classes_pred[0,1,1,1] = 0

    loss1 = handpose.loss.class_loss_mse(classes_truth, classes_pred, obj_conf, lambda_class=1)
    loss2 = handpose.loss.class_loss_mse(classes_truth, classes_pred, obj_conf, lambda_class=0.5)

    torch.testing.assert_close(loss1, torch.tensor(2.0))
    torch.testing.assert_close(loss2, torch.tensor(1.0))

def test_kpt_loss1():
    """Tests kpt_loss()"""

    kpt_truth = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
         'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
         'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
         'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
        }
    kpt_pred = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
            'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
            'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
            'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
            }
    obj_conf = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)

    loss = handpose.loss.kpt_loss(kpt_truth, kpt_pred, obj_conf)

    torch.testing.assert_close(loss, torch.tensor(0.0))

def test_kpt_loss2():
    """Tests kpt_loss()"""

    kpt_truth = {'kx_0': torch.Tensor([[0.5],[0.5]]).reshape(2,1,1,1),
         'ky_0': torch.Tensor([[0.5],[0.5]]).reshape(2,1,1,1),
         'kx_1': torch.Tensor([[0.2],[0.2]]).reshape(2,1,1,1),
         'ky_1': torch.Tensor([[0.2],[0.2]]).reshape(2,1,1,1),
        }
    kpt_pred = {'kx_0': torch.Tensor([[0.5],[0.5]]).reshape(2,1,1,1),
            'ky_0': torch.Tensor([[0.5],[0.5]]).reshape(2,1,1,1),
            'kx_1': torch.Tensor([[0.2],[0.2]]).reshape(2,1,1,1),
            'ky_1': torch.Tensor([[0.2],[0.2]]).reshape(2,1,1,1),
            }

    obj_conf = torch.Tensor([[1],[1]]).reshape(2,1,1,1)

    loss = handpose.loss.kpt_loss(kpt_truth, kpt_pred, obj_conf)

    torch.testing.assert_close(loss, torch.tensor(0.0))

def test_kpt_conf_loss1():
    """Tests kpt_conf_loss()"""

    kpt_truth = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
             'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
            }

    kpt_pred = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
                'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
                }

    obj_conf = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)

    nkpt = 2

    loss = handpose.loss.kpt_conf_loss(kpt_truth, kpt_pred, obj_conf)

    torch.testing.assert_close(loss, torch.tensor(0.0))

def test_kpt_conf_loss2():
    """Tests kpt_conf_loss()"""

    kpt_truth = {'k_conf_0': torch.Tensor([[1],[1]]).reshape(2,1,1,1),
         'k_conf_1': torch.Tensor([[1],[1]]).reshape(2,1,1,1)
        }
    kpt_pred = {'k_conf_0': torch.Tensor([[1],[1]]).reshape(2,1,1,1),
            'k_conf_1': torch.Tensor([[1],[1]]).reshape(2,1,1,1)
            }

    obj_conf = torch.Tensor([[1],[1]]).reshape(2,1,1,1)

    nkpt = 2

    loss = handpose.loss.kpt_conf_loss(kpt_truth, kpt_pred, obj_conf)

    torch.testing.assert_close(loss, torch.tensor(0.0))

def test_loss1():
    """Tests loss()"""

    x = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    y = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    w = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    h = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)

    kpt_truth = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
                'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
                'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
                'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
                }

    kpt_conf_truth = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
                'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
                }

    conf_truth = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)

    classes_truth = torch.cat([torch.Tensor([[0,0,0],[0,0,0],[0,0,0]]).reshape(1,1,3,3), 
                            torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)], dim=1)

    truth = {'x': x,
            'y': y,
            'w': w,
            'h': h,
            'conf': conf_truth,
            'kpt': kpt_truth,
            'k_conf': kpt_conf_truth,
            'classes': classes_truth
            
            }

    prediction = copy.deepcopy(truth)

    lambda_coord = 5
    lambda_noobj = 0.5
    epsilon = 1e-6
    lambda_kpt = 0.5 
    lambda_kpt_conf = 0.5

    all_losses = handpose.loss.loss_fn(truth, prediction, lambda_coord, lambda_noobj, epsilon, lambda_kpt, lambda_kpt_conf)
    
    torch.testing.assert_close(all_losses['total_loss'], torch.tensor(0.0))
    assert(all_losses['total_loss'].device == DEVICE)

def test_loss2():
    """Tests loss()"""
    x = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    y = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    w = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    h = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)

    kpt_truth = {'kx_0': torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1),
                'ky_0': torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1),
                'kx_1': torch.Tensor([[0.2], [0.2]]).reshape(2,1,1,1),
                'ky_1': torch.Tensor([[0.2], [0.2]]).reshape(2,1,1,1),
                }

    kpt_conf_truth = {'k_conf_0': torch.Tensor([[1],[1]]).reshape(2,1,1,1),
                'k_conf_1': torch.Tensor([[1],[1]]).reshape(2,1,1,1)
                }

    conf_truth = torch.Tensor([[1],[1]]).reshape(2,1,1,1)

    classes_truth = torch.cat([torch.Tensor([[0],[0]]).reshape(2,1,1,1), 
                            torch.Tensor([[1],[1]]).reshape(2,1,1,1)], dim=1)

    truth = {'x': x,
            'y': y,
            'w': w,
            'h': h,
            'conf': conf_truth,
            'kpt': kpt_truth,
            'k_conf': kpt_conf_truth,
            'classes': classes_truth
            
            }

    prediction = copy.deepcopy(truth)

    lambda_coord = 5
    lambda_noobj = 0.5
    epsilon = 1e-6
    lambda_kpt = 0.5 
    lambda_kpt_conf = 0.5

    all_losses = handpose.loss.loss_fn(truth, prediction, lambda_coord, lambda_noobj, epsilon, lambda_kpt, lambda_kpt_conf)

    torch.testing.assert_close(all_losses['total_loss'], torch.tensor(0.0))
    assert(all_losses['total_loss'].device == DEVICE)