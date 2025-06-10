import json
import os
import time
import uuid
import datetime
import logging
import json
from threading import Thread, Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from torchvision.models import vgg19, VGG19_Weights
from werkzeug.utils import secure_filename
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

# 初始化Flask应用
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

app = Flask(__name__, 
            template_folder=resource_path('templates'),
            static_folder=resource_path('static'))
CORS(app)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

# 共享状态字典及锁
tasks = {}
tasks_lock = Lock()

# 定义上传文件夹
# app.static_folder = 'static' # 已经通过 Flask 构造函数设置
CONTENT_FOLDER = 'uploads'
STYLES_FOLDER = 'styles'
HISTORY_FOLDER = 'history'
HISTORY_FILE = 'history.json'

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 设置上传文件夹和其他配置
app.config['STYLES_FOLDER'] = resource_path(STYLES_FOLDER)
app.config['CONTENT_FOLDER'] = resource_path(CONTENT_FOLDER)
app.config['HISTORY_FOLDER'] = resource_path(HISTORY_FOLDER)
app.config['MAX_HISTORY_IMAGES'] = 4  # 最多保存4个风格化历史图片
app.config['MAX_CONTENT_IMAGES'] = 3  # 最多保存3个内容图片历史
app.config['HISTORY_FILE'] = os.path.join(app.config['HISTORY_FOLDER'], HISTORY_FILE)  # 设置历史记录文件路径

# 确保所有文件夹存在
os.makedirs(app.config['CONTENT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLES_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTORY_FOLDER'], exist_ok=True)
logging.info("所有必要的文件夹已创建或已存在。")

# 加载预训练的VGG19模型到全局变量中
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
logging.info("预训练的VGG19模型已加载。")

# 风格数组（初始为空，将在初始化时加载）
style_choices = []


# 静态路由配置
@app.route('/styles/<path:filename>')
def serve_styles(filename):
    """提供风格图片"""
    return send_from_directory(app.config['STYLES_FOLDER'], filename)


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """提供上传的内容图片"""
    return send_from_directory(app.config['CONTENT_FOLDER'], filename)


@app.route('/history/<path:filename>')
def serve_history(filename):
    """提供历史记录图片"""
    return send_from_directory(app.config['HISTORY_FOLDER'], filename)


# 检查文件扩展名是否合法
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 保存历史记录
def save_history(content_img, style_img, style_weight, steps, output_img_name):
    history = {}
    if os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = {}

    # 以内容图片名为key，组织树状结构
    content_key = os.path.splitext(os.path.basename(content_img))[0]
    if content_key not in history:
        history[content_key] = []
    history[content_key].append({
        'content_img': os.path.basename(content_img),
        'style_img': os.path.basename(style_img),
        'style_weight': style_weight,
        'steps': steps,
        'output_img': f"/history/{output_img_name}"  # 确保前端可以通过URL访问
    })

    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump(history, f, indent=4)
    logging.info(f"历史记录已更新，添加了输出图像: {output_img_name}")


# 管理输出文件夹中的文件
def manage_output_folder():
    output_files = os.listdir(app.config['HISTORY_FOLDER'])
    output_files = [f for f in output_files if os.path.isfile(os.path.join(app.config['HISTORY_FOLDER'], f))]
    if len(output_files) > app.config['MAX_HISTORY_IMAGES']:
        # 删除最早的文件
        oldest_file = min(output_files, key=lambda f: os.path.getctime(os.path.join(app.config['HISTORY_FOLDER'], f)))
        os.remove(os.path.join(app.config['HISTORY_FOLDER'], oldest_file))
        logging.info(f"删除最早的历史文件: {oldest_file}")


# 更新风格选择
def update_style_choices():
    style_files = os.listdir(app.config['STYLES_FOLDER'])
    global style_choices
    style_choices = [f for f in style_files if allowed_file(f)]
    logging.info(f"已加载{len(style_choices)}个风格图片。")


# 图像加载和预处理
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # 强制图像为正方形
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3]),  # 保留RGB通道
    ])
    image = Image.open(image_name).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# 内容损失
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# 样式损失
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# 标准化
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


# 模型的内容和样式层
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0  # 索引
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'未识别的层: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # 去除未使用的层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# 输入图像的优化
def get_input_optimizer(input_img):
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)
    return optimizer


# 运行风格迁移
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       progress_callback=None, style_weight=1000000, content_weight=1):
    try:
        logging.info(f'开始风格迁移... style_weight={style_weight}, content_weight={content_weight}, steps={num_steps}')
        pbar = tqdm(desc='Processing', total=num_steps)  # 初始化进度条
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                         normalization_mean, normalization_std,
                                                                         style_img,
                                                                         content_img)
        optimizer = get_input_optimizer(input_img)

        for run in range(num_steps):
            def closure():
                input_img.data.clamp_(0, 1)  # 限制像素值
                optimizer.zero_grad()
                model(input_img)
                style_score = sum(sl.loss for sl in style_losses)
                content_score = sum(cl.loss for cl in content_losses)
                loss = style_weight * style_score + content_weight * content_score
                loss.backward()
                return loss

            optimizer.step(closure)
            input_img.data.clamp_(0, 1)  # 限制像素值
            pbar.update(1)  # 更新进度条

            # 如果提供了进度回调函数，就在每次迭代结束时调用它
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            if progress_callback:
                progress_callback(run + 1, content_score.item(), style_score.item())

            # 每隔50步或最后一步输出一次状态信息
            if (run + 1) % 50 == 0 or (run + 1) == num_steps:
                logging.info(f"步数：{run + 1} 样式损失：{style_score.item()} 内容损失：{content_score.item()}")

        pbar.close()  # 完成后关闭进度条
        return input_img
    except Exception as e:
        logging.error(f"风格迁移过程中出现错误: {e}")
        return None


# 定义风格迁移任务
def style_transfer_task(task_id, cnn, normalization_mean, normalization_std,
                        content_img_path, style_img_path, num_steps, style_weight, content_weight):
    logging.info(f"任务 {task_id} 开始处理。")
    # 加载内容和风格图片
    content_img_tensor = image_loader(content_img_path, 512 if torch.cuda.is_available() else 128)
    style_img_tensor = image_loader(style_img_path, 512 if torch.cuda.is_available() else 128)
    input_img = content_img_tensor.clone()

    # 定义进度回调函数
    def progress_callback(step, content_loss, style_loss):
        with tasks_lock:
            tasks[task_id]['progress'] = {
                'step': step,
                'content_loss': content_loss,
                'style_loss': style_loss
            }

    # 运行风格迁移
    output_img_tensor = run_style_transfer(cnn, normalization_mean, normalization_std,
                                           content_img_tensor, style_img_tensor, input_img,
                                           num_steps, progress_callback, style_weight, content_weight)
    if output_img_tensor is not None:
        # 生成输出文件名
        content_filename = os.path.splitext(os.path.basename(content_img_path))[0]
        style_filename = os.path.splitext(os.path.basename(style_img_path))[0]
        output_filename = f"{content_filename}_{style_filename}_{style_weight}_{num_steps}_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(app.config['HISTORY_FOLDER'], output_filename)
        save_image(output_img_tensor, output_path)
        save_history(content_img_path, style_img_path, style_weight, num_steps, output_filename)
        manage_output_folder()
        with tasks_lock:
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['output_path'] = f"/history/{output_filename}"
        logging.info(f"任务 {task_id} 完成，输出图像: {output_filename}")
    else:
        with tasks_lock:
            tasks[task_id]['status'] = 'failed'
        logging.error(f"任务 {task_id} 失败。")


# 保存风格迁移生成的图像
def save_image(tensor, path):
    """保存风格迁移生成的图像"""
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)
    logging.info(f"图像已保存: {path}")


# API 路由
@app.route('/')
def index():
    def getStyleName(style):
        return os.path.splitext(style)[0]

    # 确保变量名一致
    context = {
        'styleChoices': style_choices,  # 注意这里使用 styleChoices
        'getStyleName': getStyleName
    }

    logging.info("渲染主页模板。")
    return render_template('balba.html', **context)


@app.route('/api/styles')
def get_styles():
    update_style_choices()
    return jsonify(style_choices)


@app.route('/api/upload-content', methods=['POST'])
def upload_content():
    file = request.files.get('image')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['CONTENT_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"上传内容图片: {filename}")
        return jsonify({"message": "内容图片上传成功", "path": f"/uploads/{filename}"}), 200
    logging.warning("上传内容图片失败，文件类型不正确或未选择文件。")
    return jsonify({"message": "上传失败，文件类型不正确或未选择文件"}), 400


@app.route('/api/upload-style', methods=['POST'])
def upload_style():
    file = request.files.get('style_image')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['STYLES_FOLDER'], filename)
        file.save(filepath)
        update_style_choices()
        logging.info(f"上传风格图片: {filename}")
        return jsonify({"message": "风格图片上传成功", "path": f"/styles/{filename}", "styles": style_choices}), 200
    logging.warning("上传风格图片失败，文件类型不正确或未选择文件。")
    return jsonify({"message": "上传失败，文件类型不正确或未选择文件"}), 400


@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    # 从请求中获取图像和其他参数
    content_file = request.files.get('content_img')  # 获取内容图像文件
    style_file = request.files.get('style_img')  # 获取风格图像文件
    style_weight = float(request.form.get('style_weight', 10000))  # 获取风格权重
    content_weight = float(request.form.get('content_weight', 1))  # 获取内容权重
    num_steps = int(request.form.get('num_steps', 300))  # 获取迁移步数

    if not content_file or not style_file:
        logging.warning("风格迁移请求失败，缺少内容图像或风格图像。")
        return jsonify({'error': 'Content image or style image is missing'}), 400

    # 保存上传的内容图像
    content_filename = secure_filename(content_file.filename)
    content_img_path = os.path.join(app.config['CONTENT_FOLDER'], content_filename)
    content_file.save(content_img_path)

    # 保存上传的风格图像
    style_filename = secure_filename(style_file.filename)
    style_img_path = os.path.join(app.config['STYLES_FOLDER'], style_filename)
    style_file.save(style_img_path)

    # 检查图片是否存在
    if not os.path.exists(content_img_path) or not os.path.exists(style_img_path):
        logging.error("内容或风格图像未找到。")
        return jsonify({'error': 'Content or style image not found'}), 404

    # 准备保存路径将在任务完成时生成

    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    with tasks_lock:
        tasks[task_id] = {'status': 'running'}

    # 启动风格化处理任务
    thread = Thread(target=style_transfer_task, args=(
        task_id, cnn,
        torch.tensor([0.485, 0.456, 0.406]).to(device),
        torch.tensor([0.229, 0.224, 0.225]).to(device),
        content_img_path, style_img_path, num_steps, style_weight, content_weight
    ))
    thread.start()
    logging.info(f"已启动任务 {task_id}。")

    return jsonify({'task_id': task_id}), 200  # 返回 200 状态码，并返回任务ID


@app.route('/stream/<task_id>')
def stream(task_id):
    def generate():
        while True:
            with tasks_lock:
                if task_id in tasks:
                    task = tasks[task_id]
                    yield f"data: {json.dumps(task)}\n\n"
                    if task['status'] in ['completed', 'failed']:
                        break
                else:
                    yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                    break
            time.sleep(1)  # 减少CPU使用率

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/history', methods=['GET'])
def get_history():
    history_file = app.config['HISTORY_FILE']
    logging.info(f"请求历史记录文件: {history_file}")

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
                logging.info("历史记录已加载。")
                return jsonify(history), 200
            except json.JSONDecodeError as e:
                logging.error(f"加载历史记录时出错: {e}")
                return jsonify({'message': 'Error loading history data.'}), 500
    else:
        logging.info("历史记录文件不存在。")
        return jsonify({'message': 'No history available.'}), 200


@app.route('/api/check-history', methods=['GET'])
def check_history():
    history_images = os.listdir(app.config['HISTORY_FOLDER'])
    history_images = [f for f in history_images if allowed_file(f)]
    if len(history_images) >= app.config['MAX_HISTORY_IMAGES']:
        logging.info("已达到最大历史记录数量。")
        return jsonify({
            'message': 'Maximum history styles reached. Please delete an image.',
            'history_images': history_images
        }), 200
    else:
        logging.info("可以添加更多历史记录。")
        return jsonify({'message': 'You can add more styles.', 'history_images': history_images}), 200


@app.route('/api/delete-history/<key>', methods=['DELETE'])
def delete_history(key):
    # 从路径参数获取key
    content_image = key
    if not content_image:
        return jsonify({'error': '缺少内容图片参数'}), 400

    try:
        # 删除历史记录文件
        history_path = app.config['HISTORY_FILE']
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if content_image in history:
                # 删除相关的图片文件
                for item in history[content_image]:
                    output_img_path = item.get('output_img')
                    if output_img_path:
                        # 处理路径，移除前缀的/history/
                        if output_img_path.startswith('/history/'):
                            filename = output_img_path[9:]  # 移除'/history/'
                        else:
                            filename = os.path.basename(output_img_path)
                        
                        full_path = os.path.join(app.config['HISTORY_FOLDER'], filename)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            logging.info(f"删除输出图片: {full_path}")
                        else:
                            logging.warning(f"输出图片不存在: {full_path}")
                
                # 删除对应的内容图片（如果存在）
                content_img_path = os.path.join(app.config['CONTENT_FOLDER'], f"{content_image}.jpg")
                if os.path.exists(content_img_path):
                    os.remove(content_img_path)
                    logging.info(f"删除内容图片: {content_img_path}")
                
                # 检查其他可能的扩展名
                for ext in ['png', 'jpeg', 'gif']:
                    alt_content_path = os.path.join(app.config['CONTENT_FOLDER'], f"{content_image}.{ext}")
                    if os.path.exists(alt_content_path):
                        os.remove(alt_content_path)
                        logging.info(f"删除内容图片: {alt_content_path}")
                
                # 从历史记录中删除
                del history[content_image]
                
                # 保存更新后的历史记录
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                
                logging.info(f"删除历史记录: {content_image}")
                return jsonify({'message': '删除成功', 'success': True})
            else:
                return jsonify({'error': '未找到指定的历史记录'}), 404
        else:
            return jsonify({'error': '历史记录文件不存在'}), 404
    
    except Exception as e:
        logging.error(f"删除历史记录时出错: {str(e)}")
        return jsonify({'error': f'删除失败: {str(e)}'}), 500


@app.route('/api/device-info', methods=['GET'])
def get_device_info():
    """获取当前使用的设备信息"""
    try:
        device_name = str(device)
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        
        # 获取设备详细信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            device_info = {
                "device_type": device_type,
                "device_name": device_name,
                "gpu_name": gpu_name,
                "gpu_memory": f"{gpu_memory:.1f}GB",
                "description": f"使用 {gpu_name}，图像处理能力强，速度快"
            }
        else:
            device_info = {
                "device_type": device_type,
                "device_name": device_name,
                "description": "使用 CPU，图像处理能力相对较弱，速度较慢"
            }
        
        return jsonify(device_info)
    
    except Exception as e:
        logging.error(f"获取设备信息时出错: {str(e)}")
        return jsonify({'error': f'获取设备信息失败: {str(e)}'}), 500


# 定期清理过期任务
# def cleanup_tasks():
#     while True:
#         with tasks_lock:
#             now = datetime.datetime.now()
#             to_delete = []
#             for tid, t in tasks.items():
#                 if t['status'] in ['completed', 'failed']:
#                     # 假设任务完成后保留1小时
#                     completed_time = t.get('completed_at')
#                     if completed_time:
#                         elapsed = (now - completed_time).total_seconds()
#                         if elapsed > 3600:
#                             to_delete.append(tid)
#             for tid in to_delete:
#                 del tasks[tid]
#                 logging.info(f"已清理过期任务: {tid}")
#         time.sleep(600)  # 每10分钟清理一次


# 启动任务清理线程
# cleanup_thread = Thread(target=cleanup_tasks, daemon=True)
# cleanup_thread.start()
logging.info("任务清理线程已启动。")


# 更新任务完成时间
def update_task_completion(task_id):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id]['completed_at'] = datetime.datetime.now()


# 监控任务状态并更新时间
@app.after_request
def after_request(response):
    # 如果任务已完成或失败，记录完成时间
    if request.endpoint == 'stream':
        task_id = request.view_args.get('task_id')
        if task_id and task_id in tasks:
            if tasks[task_id]['status'] in ['completed', 'failed'] and 'completed_at' not in tasks[task_id]:
                update_task_completion(task_id)
    return response


if __name__ == "__main__":
    # 初始化时更新风格选择
    update_style_choices()
    logging.info("风格选择已更新。")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
