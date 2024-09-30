import cv2
import numpy as np
import gradio as gr
from scipy import ndimage
from scipy.interpolate import interp1d

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    height, width , _= image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    x, y = np.meshgrid(gridX, gridY)
    
    p, q = target_pts[:, [1, 0]], source_pts[:, [1, 0]]
    num = len(target_pts)

    # 为加快运行速度，参考变换维数，全部使用矩阵操作代替循环
    reshaped_p = p.reshape(num, 2, 1, 1)
    reshaped_q = q.reshape((num, 2, 1, 1))  
    reshaped_v = np.stack((y, x), axis=0)
    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha
    w /= np.sum(w, axis=0, keepdims=True)

    pstar = np.sum(w[:, np.newaxis] * reshaped_p, axis=0)
    qstar = np.sum(w[:, np.newaxis] * reshaped_q, axis=0)
    phat = reshaped_p - pstar
    qhat = reshaped_q - qstar 

    vpstar = reshaped_v - pstar    
    reshaped_vpstar = vpstar.reshape(2, 1, height, width)
    neg_vpstar_verti = vpstar[[1, 0],...]                 
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, height, width)
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)
    reshaped_mul_right = mul_right.reshape(2, 2, height, width)                                                
    
    sum1 = np.zeros((height, width, 2), np.float32)
    for i in range(num):                                          
        p_hat = phat[i].reshape(1, 2, height, width)                          
        w_new = w[i].reshape(1, 1, height, width)                                 
        neg_phat_verti = phat[i][[1, 0]]                                                  
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, height, width)           
        mul_left = np.concatenate((p_hat, reshaped_neg_phat_verti), axis=0)   
        
        A = np.matmul((w_new * mul_left).transpose(2, 3, 0, 1), 
                       reshaped_mul_right.transpose(2, 3, 0, 1))                      
                                                
        reshaped_qhat = qhat[i].reshape(1, 2, height, width).transpose(2, 3, 0, 1)   
        sum1 += np.matmul(reshaped_qhat, A).reshape(height, width, 2)     
                                                        
    sum2 = np.linalg.norm(sum1.transpose(2, 0, 1), axis=0, keepdims=True)              
    transformers = (sum1.transpose(2,0,1)) / sum2 * (np.linalg.norm(vpstar, axis=0, keepdims=True))  + qstar                       

    # Flatten the mask and find indices of NaN and non-NaN values
    nan_mask = sum2[0] == 0
    nan_mask_flat = np.flatnonzero(nan_mask)
    nan_mask_anti_flat = np.flatnonzero(~nan_mask)

    # Interpolation function for each transformer
    interp_func_0 = interp1d(nan_mask_anti_flat, transformers[0][~nan_mask], kind='linear', fill_value="extrapolate")
    interp_func_1 = interp1d(nan_mask_anti_flat, transformers[1][~nan_mask], kind='linear', fill_value="extrapolate")

    # Replace NaN values with interpolated values
    transformers[0][nan_mask] = interp_func_0(nan_mask_flat)
    transformers[1][nan_mask] = interp_func_1(nan_mask_flat)

    # Remove the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    warped_image[y, x] = image[tuple(transformers.astype(np.int16))]     
    
    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
