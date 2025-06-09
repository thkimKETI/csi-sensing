# Standard Library
from io import StringIO
import sys, os
import csv
import json
import re
import argparse
import datetime
import multiprocessing
import time
import warnings
import queue
from multiprocessing import Process, Queue, Value, Array, Lock, Manager
from threading import Thread
import ast

# Third-party
from tqdm import tqdm
import serial
import numpy as np
import pandas as pd
import torch

# GUI library
from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import pyqtgraph as pg

# MODEL
from models import *

# butterworth library
from scipy.signal import butter, filtfilt

# MQTT Library
from mqtt_config import BROKER_ADDRESS, PORT, TOPIC, MAC_ADDRESS, create_mqtt_message
import paho.mqtt.client as mqtt
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ======================================================= #
def parse_argument():
    parser = argparse.ArgumentParser(description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-s', '--inf_sec', dest='inf_sec', type=int, required=False, default=3) # 
    parser.add_argument('-m', '--model', dest='model', type=str, required=False, default='CNN')
    parser.add_argument('-a', '--acquire', dest='acq', action='store_true', required=False, default=False)
    parser.add_argument('-d', '--dir', dest='dir', type=str, required=False, default=datetime.datetime.now().strftime("%m%d"))
    parser.add_argument('-pr', '--prev_sec', dest='prev_sec', type=int, required=False, default= 3) 
    args = parser.parse_args()  
    return args.acq, args.dir, args.model, args.inf_sec, args.prev_sec

acq_bool, csi_dir, model_type, inf_sec, prev_sec = parse_argument()

SEQUENCE_LENGTH = 180  # sequence length of time series data
sequence_len_inf = SEQUENCE_LENGTH * inf_sec
sequence_prev_inf = SEQUENCE_LENGTH * prev_sec

CSI_SAVE_PATH = f"/csi/datasets/{csi_dir}" # CSI DATA 저장 경로 ex. 0409, 0410
os.makedirs(CSI_SAVE_PATH, exist_ok=True)

# ======================================================= #
#              공유 자원 및 큐 설정                        #
# ======================================================= #
# 멀티프로세싱 관리자 설정
manager = Manager()
LABELS = manager.dict({"time": "", "occ": "", "loc": "", "act": ""})

# MAC 주소 리스트 초기화
mac_list = MAC_ADDRESS

# 데이터 취득 상태 관리
isPushedBtn = Value('b', False)
GET_START_FLAG = Value('b', True)  # 데이터가 1초안에 Sequence-len 만큼 쌓였는지 확인하는 Flag

# 포트별 가비지 카운터
garbage_counters = [Value('i', 0) for _ in range(4)]

# 데이터 큐 설정 - 프로세스 간 데이터 전송을 위한 큐
data_queues = [Queue(maxsize=180) for _ in range(4)]  # 각 ESP 디바이스별 데이터 큐
inference_queue = Queue(maxsize=180)  # 추론을 위한 데이터 큐
storage_queue = Queue(maxsize=180) # 데이터를 저장하기 위한 큐 
visualization_queues = [Queue(maxsize=180) for _ in range(4)]  # 시각화를 위한 데이터 큐


# 시각화를 위한 공유 배열 (락 메커니즘 사용)
csi_bt_array_shared = [Array('d', 180 * 114) for _ in range(4)]
csi_raw_array_shared = [Array('d', 180 * 192) for _ in range(4)]
csi_fg_array_shared = [Array('d', 180 * 114) for _ in range(4)]
locks = [Lock() for _ in range(4)]

# 추론을 위한 공유 배열
inference_base = Array('f', 180 * 114 * 4, lock=True)
inference_array = np.frombuffer(inference_base.get_obj(), dtype=np.float32).reshape((180, 114, 4))
# 포트별 현재 인덱스 및 준비 상태 플래그
port_index = Array('i', [0, 0, 0, 0])
ready_flags = Array('i', [0, 0, 0, 0])

# 프로세스 종료 플래그
exit_flag = Value('b', False)

# 프로세스 리스트 (종료 시 사용)
PROCESSES = []

# 데이터 컬럼 갯수 비교용
DATA_COLUMNS_NAMES = ["type", "id", "timestamp", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

# ======================================================= #
#                   유틸리티 함수                          #
# ======================================================= #
def get_amplitude_visual(csi, is_sequence=False):
    csi = np.array(csi)

    # Zero-subcarriers 제거 및 Roll subcarrier index 
    # csi = np.concatenate((csi[254:368],csi[132:246]),axis=0)

    if is_sequence==True: 
        even_elements = csi[:,::2]
        odd_elements = csi[:,1::2]
    else:
        even_elements = csi[::2]
        odd_elements = csi[1::2]
    amplitude = np.sqrt(np.square(even_elements) + np.square(odd_elements))

    # min-max 정규화 
    # return (amplitude-min(amplitude)) /(max(amplitude)-min(amplitude)) 
    return amplitude / 15.0

def get_amplitude(csi, is_sequence=False):
    csi = np.array(csi)

    # Zero-subcarriers 제거 및 Roll subcarrier index 
    csi = np.concatenate((csi[254:368],csi[132:246]),axis=0)

    if is_sequence==True: 
        even_elements = csi[:,::2]
        odd_elements = csi[:,1::2]
    else:
        even_elements = csi[::2]
        odd_elements = csi[1::2]
    amplitude = np.sqrt(np.square(even_elements) + np.square(odd_elements))

    # min-max 정규화 
    # return (amplitude-min(amplitude)) /(max(amplitude)-min(amplitude)) 
    return amplitude / 15.0

def butterworth_filter(data, cutoff, fs, order, filter_type='low', prev_data=None):
    """
    Butterworth 필터 적용 함수
    
    Args:
        data: 필터링할 데이터
        cutoff: 차단 주파수
        fs: 샘플링 주파수
        order: 필터 차수
        filter_type: 필터 유형 ('low', 'high', 'band', 'bandstop')
        prev_data: 이전 데이터 (연속성을 위해 사용)
    
    Returns:
        filtered_data: 필터링된 데이터
    """
    nyquist = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = filtfilt(b, a, data, axis=0) # 필터 적용
    filtered_data = np.ascontiguousarray(filtered_data) # 음수 스트라이드를 방지하기 위해 복사
    return filtered_data

def robust_pca(M, lamb=None, mu=None, max_iter=1, tol=1e-5):
    """
    ### Robust Principal Component Analysis (RPCA) 구현
    
    Args:
        M: 입력 행렬
        lamb: 정규화 파라미터
        mu: 증강 라그랑지안 파라미터
        max_iter: 최대 반복 횟수
        tol: 수렴 허용 오차
    
    Returns:
        L: 저차원 행렬 (배경)
        S: 희소 행렬 (전경)
    """
    # 행렬 노름 계산
    norm_M = np.linalg.norm(M, ord='fro')
    
    # 초기화
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    
    # 파라미터 설정
    if lamb is None:
        lamb = 1.0 / np.sqrt(max(M.shape))
    if mu is None:
        mu = M.size / (4.0 * np.linalg.norm(M, ord=1))
    
    # 초기 Y 설정
    Y = M / max(np.linalg.norm(M, ord=2), np.linalg.norm(M, ord=np.inf) / lamb)
    
    # RPCA 알고리즘 반복
    for _ in range(max_iter):
        # L 업데이트 (특이값 임계처리)
        U, sigma, VT = svd(M - S + Y / mu, full_matrices=False)
        sigma_shrink = np.maximum(sigma - 1.0 / mu, 0)
        L_new = U @ np.diag(sigma_shrink) @ VT
        
        # S 업데이트 (소프트 임계처리)
        S_new = np.sign(M - L_new + Y / mu) * np.maximum(np.abs(M - L_new + Y / mu) - lamb / mu, 0)
        
        # Y 업데이트 (라그랑지안 승수)
        Y = Y + mu * (M - L_new - S_new)
        
        # 수렴 확인
        PL = L_new - L
        PS = S_new - S
        S = S_new
        L = L_new
        
        err = np.linalg.norm(PL, 'fro') / norm_M + np.linalg.norm(PS, 'fro') / norm_M
        if err < tol:
            break
    
    return L, S

def apply_chunk_butterworth_and_rpca(data_chunk, fs=60, cutoff=0.1, order=2, rpca_lamb=None):
    """
    데이터 청크에 Butterworth 필터와 RPCA를 순차적으로 적용
    
    Args:
        data_chunk: 처리할 데이터 청크 (2D 배열: 시간 x 특성)
        fs: 샘플링 주파수
        cutoff: 차단 주파수
        order: 필터 차수
        rpca_lamb: RPCA 정규화 파라미터
    
    Returns:
        filtered_data: Butterworth 필터링된 데이터
        background: RPCA로 추출한 배경 (저차원 행렬 L)
        foreground: RPCA로 추출한 전경 (희소 행렬 S)
    """
    # 데이터가 충분한지 확인
    if data_chunk.shape[0] < 10:
        return data_chunk, data_chunk, np.zeros_like(data_chunk)
    
    # 1. Butterworth 필터 적용
    filtered_data = butterworth_filter(data_chunk, cutoff=cutoff, fs=fs, order=order)
    
    # 2. RPCA 적용
    #L, S = robust_pca(filtered_data, lamb=rpca_lamb)
    #U, Sigma, VT = randomized_svd(filtered_data, n_components=1, random_state=7671)
    #rank1 = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    #avg_vector = np.mean(rank1, axis=0)
    #static1 = np.tile(avg_vector, (data_chunk.shape[0],1))

    rank1 = filtered_data

    # 2. inference_len 크기의 평균 벡터 추출 및 크기 확장
    avg_vec = np.mean(filtered_data,axis=0) 
    static1 = np.tile(avg_vec, (data_chunk.shape[0],1))

    return filtered_data, rank1, static1

def terminate_process():
    """모든 프로세스를 종료하는 함수"""
    global PROCESSES, exit_flag
    
    print("[🛑] 모든 프로세스를 종료합니다...")
    exit_flag.value = True
    
    for process in PROCESSES:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    
    print("[✅] 모든 프로세스가 종료되었습니다.")

# ======================================================= #
#                   MQTT 콜백 함수                          #
# ======================================================= #
def on_connect(client, userdata, flags, rc):
    """MQTT 브로커 연결 시 호출될 콜백 함수"""
    # print(f"[on_connect] call, rc = {rc}")
    if rc == 0:
        # print(f"MQTT 브로커 연결 성공: {BROKER_ADDRESS}:{PORT}")
        try:
            result, mid = client.subscribe(TOPIC)
            if result == mqtt.MQTT_ERR_SUCCESS:
                pass
            else:
                print(f"[❌] fail topic subscribe: {TOPIC}, result={result}")
        except Exception as e:
            print(f"오류: 토픽 구독 실패: {e}")
    else:
        print(f"[❌] MQTT 연결 실패: 코드 {rc}. 브로커 주소 및 포트를 확인하세요.")

def on_disconnect(client, userdata, rc):
    """MQTT 브로커 연결 해제 시 호출될 콜백 함수"""
    print(f"[🔌] MQTT 브로커 연결 해제됨: 코드 {rc}")
    if rc != 0:
        print("[❌] 예상치 못한 연결 끊김. 재연결 시도 안 함 (필요시 재연결 로직 추가).")
    # stop_event.set() # 연결 해제 시 항상 종료할 필요는 없을 수 있음


from collections import defaultdict

mac_current_second = defaultdict(lambda: None)
mac_second_count = defaultdict(lambda: 0)

def on_message_with_queue(data_queue):
    global mac_current_second, mac_second_count
    """MQTT 메세지 수신 시 호출될 콜백 함수"""
    global DATA_COLUMNS_NAMES
    def _on_message(client, userdata, msg):
        try:
            payload_str = msg.payload.decode("utf-8")
            parts = payload_str.split(',',4)

            group_mac = parts[0] # mac_address : string type
            # print(group_mac)

            # group_ntp_millis = int(parts[1]) # 그룹 전체의 NTP 시간 (참고용)
            # data_format = parts[2] # 필요시 사용
            # group_count_header = int(parts[3]) # 헤더의 개수, 필요시 사용
            entries = re.findall(r'\{CSI_DATA,[^}]+\}', parts[4])
            for entry in entries:
                # csi_data_read_parse Preprocess #
                strings = entry.lstrip('{').rstrip('\\r\\n\'}')
                index = strings.find('CSI_DATA')

                if index != 0:
                    # 'CSI_DATA' 문자열이 맨 앞에 없으면 무시
                    continue

                csv_reader = csv.reader(StringIO(strings)) # Str to CSV
                csi_data = next(csv_reader) # CSV to List
                csi_data_len = int (csi_data[-3])


                if len(csi_data) != len(DATA_COLUMNS_NAMES):
                    # 데이터의 컬럼 개수가 기대하는 컬럼 개수와 다르면 무시
                    print(f"[⛔️ ] 데이터 컬럼 개수({len(csi_data)})가 기대하는 컬럼 개수({len(DATA_COLUMNS_NAMES)})와 다릅니다.")
                    continue

                try:
                    # 마지막 컬럼(csi_data 필드)가 JSON이 아니거나 파싱에 실패하게 되면, 잘못된 데이터이므로 무시
                    csi_raw_data = json.loads(csi_data[-1])
                    timestamp = csi_data[2] # CSI 데이터 생성시 시간
                except:
                    print(f"[⛔️ ] 마지막 컬럼(CSI 데이터 필드)가 JSON 형식이 아닙니다.")
                    continue
		
                # if csi_data_len != len(csi_raw_data):
                #     # csi_data의 -3번째 필드(데이터 길이)와 실제 데이터의 길이가 다르면, 잘못된 데이터이므로 무시
                #     print(f"[⛔️ ] csi_data의 -3번째 필드({csi_data_len}가 실제 데이터의 길이({len(csi_raw_data)})와 다릅니다. ")
                #     print(csi_data)
                #     continue

                # timestamp에서 초 단위 추출 (ex: '10:42:12.039' → 12)
                ts_second = int(timestamp.split(":")[2].split(".")[0])
                mac_idx = mac_list.index(group_mac) # mac_address -> mac_idx (0, 1)

                # 처음 수신 or 초가 바뀌면 출력 후 초기화
                if mac_current_second[mac_idx] is None:
                    mac_current_second[mac_idx] = ts_second

                # if ts_second != mac_current_second[mac_idx]:
                #     present = time.time()
                #     present_str = time.strftime("%H:%M:%S", time.localtime(present))
                #     print(f"csi per Second[{mac_idx}] - 초당 데이터 개수: {mac_second_count[mac_idx]} - esp_time: {timestamp} - arrive_time: {present_str}")
                    
                #     # 초기화
                #     mac_current_second[mac_idx] = ts_second
                #     mac_second_count[mac_idx] = 0

                # count 증가
                mac_second_count[mac_idx] += 1
                data_queue.put((mac_idx, timestamp, csi_raw_data))
                
                # if len(float_list) != 384:
                #     print("Not 384!", len(float_list))


                # 4. (타임스탬프(str), CSI 데이터(list)) 튜플로 Queue에 넣기
                # data_queue.put((mac_list.index(group_mac), timestamp, float_list))


        except UnicodeDecodeError:
            print(f"오류: 메시지 디코딩 실패 (UTF-8). Topic: {msg.topic}")
        except Exception as e:
            print(f"오류: on_message 콜백 처리 중 예외 발생: {e}")
    return _on_message

# ======================================================= #
#                   모델 로딩 함수                         #
# ======================================================= #
def load_model(path, n_classes, model_type):

    device = "cpu"  # 필요시 CUDA 사용 가능
    
    #path = path.split("/")
    if model_type == "CNN":
        model = WiFiCSICNNAttention(num_classes=n_classes, num_esp=2)
        #path[-2] = "CNN"
    else:
        model = Transformer(
            feature=114,
            d_model=64,
            n_head=4,
            max_len=(SEQUENCE_LENGTH-10) * 2,
            ffn_hidden=32,
            n_layers=2,
            drop_prob=0.1,
            n_classes=n_classes,
            device=device).to(device=device)
        path[-2] = "Transformer"

    #path = "/".join(path)   
    #print(path)   
    #model.load_state_dict(torch.load(path, weights_only=True))

    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model.to(device)

# ======================================================= #
#                 MQTT 구독+처리 프로세스                      #
# ======================================================= #
def data_mqttsub_process(data_queue, garbage_counter, exit_flag):
    """MQTT CSI Data Subscriber

    Connects to an MQTT broker, subscribes to a specified topic, and stores received CSI data in a data-queue.

    Args:
        broker_address: MQTT 브로커의 IP 주소 또는 호스트명.
        topic: 구독할 MQTT 토픽명 (예: 'csi/data').        
        data_queue: 취득한 데이터를 전송할 큐
        exit_flag: 프로세스 종료 플래그
    """
    # MQTT Settings
    client_id = f"csi-subscriber-{os.getpid()}"
    client = mqtt.Client(client_id=client_id)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        client.connect(BROKER_ADDRESS, PORT, 60)
        print(f"[✅] MQTT 브로커({BROKER_ADDRESS}:{PORT}) 연결 성공, TOPIC:{TOPIC}")
        client.loop_start()
        while not exit_flag.value:
            try:
                if isPushedBtn.value:
                    client.on_message = on_message_with_queue(data_queue) # 실질적으로 데이터를 큐에 저장하는 역할
                    time.sleep(0.00001)
            except Exception as e:
                print(f"[❌] MQTT 데이터 구독 오류: {e}")
                time.sleep(1)  # 오류 발생 시 잠시 대기

    except Exception as e:
        print(f"[❌] MQTT 브로커 연결 실패: {e}")

def data_mqtt_processing_process(data_queue, inference_queue, storage_queue, visualization_queue, exit_flag):
    """MQTT CSI Data Subscriber processing
    Processes incoming CSI (Channel State Information) data, applies filtering, and prepares it for visualization and storage.

    Args:
        data_queue (Queue): Queue receiving raw CSI data from the MQTT listener.
        inference_queue (Queue) : Queue used to recognize Location / Action
        storage_queue (Queue): Queue used to pass processed CSI data to storage components (e.g., file writer, database).
        visualization_queue (Queue): Queue used to pass processed CSI data for real-time visualization.
        exit_flag (Value): A multiprocessing.Value object used to signal termination of the process loop.

    """
    print(f"[🔄] MQTT 브로커에 구독한 데이터 처리 프로세스 시작")    

    # 데이터 처리를 위한 버퍼 초기화
    buffer_size = 180  # 버퍼 크기
    raw_buffers = {}
    bt_buffers = {}
    fg_buffers = {}

    # 배경 데이터 초기화 (정적 상태 데이터)
    background_data = None
    background_counter = 0
    background_samples = 100  # 배경으로 사용할 샘플 수                    

    # 추론용 데이터 버퍼
    inference_buffers = {}
    second_cnt = [0,0]
    current_ts = 0
    error_cnt = [0,0]
    # 데이터 처리 루프
    buffer_index = 0
    while not exit_flag.value:
        try:
            # 데이터 큐에서 CSI 데이터 가져오기 (비차단 방식)
            try:
                mac, ts, csi_data = data_queue.get_nowait()

                # timestamp -> 초 단위로 변환
                ts_second = int(ts.split(":")[2].split(".")[0])
                # print(ts_second)

                # if (csi_data[0] != 0) or (csi_data[247] != 0) or (csi_data[373] != 0) or (csi_data[128] == 0):
                #     error_cnt[mac] +=1
                #     continue

                # if any(val == 0 or val > 88 for val in csi_data[132:142]):
                #     error_cnt[mac] +=1
                #     # print("ERROR",csi_data[132:142])
                #     continue


                # 새로운 초로 넘어갔을 때 출력 및 초기화
                # if current_ts is None:
                #     current_ts = ts_second

                # if ts_second != current_ts:
                #     # 출력
                #     print(f"csi per Second[0] \n 에러 발생 횟수 : {error_cnt[0]} \n 초당 데이터 개수 : {second_cnt[0]} \n 시간 : {ts}")
                #     print(f"csi per Second[1] \n 에러 발생 횟수 : {error_cnt[1]} \n 초당 데이터 개수 : {second_cnt[1]} \n 시간 : {ts}")
                    
                #     # 현재 초 갱신
                #     current_ts = ts_second

                #     # 카운트 초기화
                #     second_cnt = [0, 0]
                #     error_cnt = [0, 0]

                # # 카운트 업데이트
                # second_cnt[mac] += 1
                
            except queue.Empty:
                time.sleep(0.001)  # 큐가 비어있으면 잠시 대기
                continue
            
            # MAC 주소별 버퍼 초기화
            if mac not in raw_buffers:
                raw_buffers[mac] = np.zeros([buffer_size, 192])
            if mac not in bt_buffers:
                bt_buffers[mac] = np.zeros([buffer_size, 114])
            if mac not in fg_buffers:
                fg_buffers[mac] = np.zeros([buffer_size, 114])
            if mac not in inference_buffers:
                inference_buffers[mac] = []

            # CSI 데이터 처리
            amplitude = get_amplitude(csi_data) # test chase : confirm zero subcarrier!
            amplitude_visual = get_amplitude_visual(csi_data)
            # 버퍼에 원시 데이터 저장
            raw_buffers[mac][buffer_index] = amplitude_visual

            # 추론용 데이터 버퍼에 추가
            inference_buffers[mac].append(amplitude)
            if len(inference_buffers[mac]) > buffer_size:
                print("dklsjkkdsjfksajkjdfkaf")
                inference_buffers[mac].pop(0)
            
            
            # 추론 데이터 크기만큼의 chunk가 모이면 Butterworth 필터와 RPCA 적용
            if len(inference_buffers[mac]) == buffer_size:
                # 데이터 청크 준비
                data_chunk = np.array(inference_buffers[mac])
                inference_buffers[mac].pop(0)
                #inference_buffer = []      #sehora
                #print(data_chunk.shape)
                
                # Butterworth 필터와 RPCA 적용
                filtered_chunk, full_rank1, static1 = apply_chunk_butterworth_and_rpca(
                    data_chunk, 
                    fs=2, 
                    cutoff=0.1, 
                    order=1
                )
                
                # 시각화용 버퍼 사이즈가 추론 사이즈 보다 클 경우, 시각화용 버퍼 업데이트 (가장 최근 데이터)
                #bt_buffer[buffer_index] = filtered_chunk[-1]
                #fg_buffer[buffer_index] = (filtered_chunk[-1] - static1[-1])*2.0
                # 시각화용 버퍼 사이즈와 추론 사이즈가 일치할 경우, 시각화용 버퍼 업데이트
                bt_buffers[mac] = filtered_chunk
                fg_buffers[mac] = (filtered_chunk - static1) * 2.0
                #bt_buffers[mac] = data_chunk
                #fg_buffers[mac] = data_chunk

                
                # 추론 및 저장을 위한 충분한 데이터가 수집되었고 버튼이 눌려있을 때
                if isPushedBtn.value:
                    # 추론 큐가 가득 차면 오래된 데이터 제거
                    # if inference_queue.full():
                    #     try:
                    #         inference_queue.get_nowait()
                    #     except queue.Empty:
                    #         pass
                    
                    if storage_queue.full():
                        try: 
                            storage_queue.get_nowait()
                        except queue.Empty:
                            pass

                    # # 추론 큐에 데이터 추가 (필터링된 데이터 사용)
                    inference_queue.put((mac, fg_buffers[mac]))

                    # 데이터 저장 큐에 데이터 추가 (amplitude 데이터 사용)
                    storage_queue.put((mac, data_chunk))
            else:
                # 충분한 데이터가 없을 때는 원시 데이터 사용
                bt_buffers[mac][buffer_index] = amplitude
                fg_buffers[mac][buffer_index] = np.zeros_like(amplitude)
            
            #---------------------------------------------------------------------------------
            # 시각화를 위한 데이터 업데이트
            with csi_raw_array_shared[mac].get_lock():
                raw_array = np.frombuffer(csi_raw_array_shared[mac].get_obj(), dtype=np.float64).reshape(180, 192)
                raw_array[:] = raw_buffers[mac]
            
            with csi_bt_array_shared[mac].get_lock():
                bt_array = np.frombuffer(csi_bt_array_shared[mac].get_obj(), dtype=np.float64).reshape(180, 114)
                bt_array[:] = bt_buffers[mac]
            
            with csi_fg_array_shared[mac].get_lock():
                fg_array = np.frombuffer(csi_fg_array_shared[mac].get_obj(), dtype=np.float64).reshape(180, 114)
                fg_array[:] = fg_buffers[mac]

            # 버퍼 인덱스 업데이트 (순환 버퍼)
            buffer_index = (buffer_index + 1) % buffer_size
            
        except Exception as e:
            print(f"[❌] MQTT 구독 데이터 처리 오류: {e}")
            time.sleep(0.1)  # 오류 발생 시 잠시 대기
    
    print(f"[🔄] MQTT 구독 데이터 처리 프로세스 종료")

def parse_csi_data(line):
    """
    Extracts the CSI values from a CSI_DATA line.
    Returns a list of floats.
    """
    try:
        # Find the start of the quoted list (first field starting with "[")
        quote_start = line.find('["')
        if quote_start == -1:
            quote_start = line.find('"[')
        if quote_start == -1:
            raise ValueError("CSI data not found in line.")

        # Grab everything from that quote onward
        csi_str = line[quote_start:].strip()

        # Remove surrounding quotes
        if csi_str.startswith('"') and csi_str.endswith('"'):
            csi_str = csi_str[1:-1]

        # Now safely evaluate the list
        csi_values = ast.literal_eval(csi_str)

        # Convert all to float
        return [float(x) for x in csi_values]

    except Exception as e:
        print(f"Error parsing CSI data: {e}")
        return None

# ======================================================= #
#                 신경망 추론 프로세스                        #
# ======================================================= #
def neural_network_inference_process(inference_queue, storage_queue, labels_dict, exit_flag, acq_bool):
    """
    신경망 추론을 수행하는 프로세스
    
    Args:
        inference_queue: 추론할 데이터를 받을 큐
        labels_dict: 추론 결과를 저장할 공유 딕셔너리
        exit_flag: 프로세스 종료 플래그
    """
    print("[🧠] 신경망 추론 프로세스 시작")
    
    # #MQTT 클라이언트 설정
    # client = mqtt.Client()
    # try:
    #     client.connect(BROKER_ADDRESS, PORT)
    #     print("[✅] MQTT 브로커 연결 성공")
    # except Exception as e:
    #     print(f"[❌] MQTT 브로커 연결 실패: {e}")
    
    # 모델 로딩 (Load Model)
    device = "cpu"  # 필요시 CUDA 사용 가능
    model_name = "0401_0409_60_1S_PREV"
    save_cnt = 0
    
    try:
        # model_occ = load_model(f"/csi/weight/{model_name}/CNN/occ", n_classes=2, model_type=model_type)
        model_loc = load_model("/csi/weight/esp01_weight/loc.pt", n_classes=4, model_type="CNN")
        model_act = load_model("/csi/weight/esp01_weight/act.pt", n_classes=4, model_type="CNN")
        print("[✅] 모델 로딩 성공")
    except Exception as e:
        print(f"[❌] 모델 로딩 실패: {e}")
        return
    
    # 클래스 매핑
    loc_classes = ["Z0", "Z1", "Z2", "Z3"]
    act_classes = ["Exr", "Sit", "Stand", "Walk"]
    gst_classes = ["circle", "line"]

    start_time = time.time()
    inf_flag = False
    
    # 추론 루프
    inference_data = {i: None for i in range(2)}  # 각 포트별 추론 데이터
    storage_data = {i: None for i in range(2)} # 각 포트별 저장 데이터
    while not exit_flag.value:

        try:
            # 추론 큐에서 데이터 가져오기 (비차단 방식)
            try:
                port_num, data = inference_queue.get_nowait()
                inference_data[port_num] = data

                s_port_num, s_data = storage_queue.get_nowait()
                # print(port_num, s_port_num, data, "\n",  s_data)
                storage_data[s_port_num] = s_data
                
            except queue.Empty:
                time.sleep(0.01)  # 큐가 비어있으면 잠시 대기
                continue

            if time.time() - start_time > 0.5: 
                inf_flag = True
            
            
            # MQTT Data save
            # if acq_bool:
            #     if save_cnt <= 360:
            #             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 날짜_시간 (초 단위까지)
            #             os.makedirs("/csi/datasets/" +timestamp[4:8], exist_ok=True)
            #             filename = f"/csi/datasets/{timestamp[4:8]}/{timestamp}_{save_cnt}.csv"
            #             np.savetxt(filename, storage_data[0], delimiter=",")
            #             save_cnt += 1
            #             print(f"# of saved files: {save_cnt}")                    

            
            # 모든 포트에서 데이터가 준비되었는지 확인
            if all(data is not None for data in inference_data.values()) and isPushedBtn.value and inf_flag:
                # 데이터 준비
                # Reconstruction Error 확인
                #print("------------------------------------------------")
                #print(int(sum(sum(abs(inference_data[0])))),int(sum(sum(abs(inference_data[1])))),int(sum(sum(abs(inference_data[2])))),int(sum(sum(abs(inference_data[3])))) )
                # 데이터 저장
                if acq_bool: 
                    save_data = np.concatenate([storage_data[0], storage_data[1]], axis=0)
                    if save_cnt <= 360:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 날짜_시간 (초 단위까지)
                            os.makedirs("/csi/datasets/" +timestamp[4:8], exist_ok=True)
                            filename = f"/csi/datasets/{timestamp[4:8]}/{timestamp}_{save_cnt}.csv"
                            np.savetxt(filename, save_data, delimiter=",")
                            save_cnt += 1
                            print(f"# of saved files: {save_cnt}")
                    else: 
                        print("🍀🍀🍀🍀🍀🍀🍀🍀🍀🍀End of saving🍀🍀🍀🍀🍀🍀🍀🍀🍀🍀🍀")
                
                combined_data = np.stack([inference_data[i] for i in range(2)], axis=-1)
                tensor_data = torch.tensor(combined_data, dtype=torch.float32).unsqueeze(0).to(device)
                tensor_data = tensor_data.permute(0, 3, 1, 2)



                # 1s, 2s 
                #tensor_data = tensor_data[:, :, -60:, :]
                
                # 추론 수행
                with torch.no_grad():
                    # 병렬 추론을 위한 스레드 생성
                    loc_result = [None]
                    act_result = [None]
                    
                    def infer_loc():
                        result = model_loc(tensor_data)
                        _, loc_result[0] = torch.max(result, 1)
                    
                    def infer_act():
                        result = model_act(tensor_data)
                        _, act_result[0] = torch.max(result, 1)
                    
                    # 병렬 추론 실행
                    t1 = Thread(target=infer_loc)
                    t2 = Thread(target=infer_act)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                
                # 결과 처리
                #loc_pred = torch.argmax(loc_output, dim=1).item(
                #act_pred = torch.argmax(act_output, dim=1).item()
                
                # 결과 저장
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                labels_dict["time"] = current_time
                labels_dict["loc"] = loc_classes[loc_result[0].item()]
                labels_dict["act"] = act_classes[act_result[0].item()]
                
                # # MQTT로 결과 전송
                # try:
                #    message = create_mqtt_message(
                #        time=current_time,
                #        loc=labels_dict["loc"],
                #        act=labels_dict["act"]
                #     )
                #    client.publish(TOPIC, message)
                # except Exception as e:
                #    print(f"[❌] MQTT 메시지 전송 실패: {e}")
                
                # 추론 데이터 초기화 (다음 추론을 위해)
                # print(inference_data, np.shape(inference_data))
                for i in range(2):
                    inference_data[i] = None
                #time.sleep(0.5)

                start_time = time.time()
                inf_flag = False
        
        except Exception as e:
            import traceback
            print(f"[❌] 신경망 추론 오류: {e}")
            traceback.print_exc()
            time.sleep(0.0001)  # 오류 발생 시 잠시 대기
    
    print("[🧠] 신경망 추론 프로세스 종료")

# ======================================================= #
#                      GUI 클래스                          #
# ======================================================= #
class CSIDataGraphicalWindow(QMainWindow):
    def __init__(self, labels_dict, isPushedBtn):
        super().__init__()
        
        self.labels_dict = labels_dict
        self.isPushedBtn = isPushedBtn
        
        self.setWindowTitle("Quad-chips CSI SENSING")
        self.setGeometry(1500, 0, 1600, 1400) # location(x, y), width, height

        # SETTING MAIN WIDGET & LAYOUT
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.layout = QVBoxLayout()
        self.mainWidget.setLayout(self.layout)

        # SETTING PYQTGRAPH
        self.graphWidget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphWidget)  # 그래프 위젯 추가

        self.plotItem_p1 = self.graphWidget.addPlot(row=0, col=0, title="PORT 1") 
        self.plotItem_p1.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p0 = self.graphWidget.addPlot(row=0, col=1, title="PORT 0") 
        self.plotItem_p0.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p1_fg = self.graphWidget.addPlot(row=0, col=2, title="PORT 1 (Foreground : Origin - R1)") 
        self.plotItem_p1_fg.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p0_fg = self.graphWidget.addPlot(row=0, col=3, title="PORT 0 (Foreground : Origin - R1)") 
        self.plotItem_p0_fg.setLabels(left='CSI Carrier Number', bottom='Time')

        self.plotItem_p3 = self.graphWidget.addPlot(row=1, col=0, title="PORT 3") 
        self.plotItem_p3.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p2 = self.graphWidget.addPlot(row=1, col=1, title="PORT 2") 
        self.plotItem_p2.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p3_fg = self.graphWidget.addPlot(row=1, col=2, title="PORT 3 (Foreground : Origin - R1)") 
        self.plotItem_p3_fg.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p2_fg = self.graphWidget.addPlot(row=1, col=3, title="PORT 2 (Foreground : Origin - R1)") 
        self.plotItem_p2_fg.setLabels(left='CSI Carrier Number', bottom='Time')

        # SETTING HEATMAP
        self.heatmap_p1 = pg.ImageItem(border='w')
        self.plotItem_p1.addItem(self.heatmap_p1)
        self.heatmap_p0 = pg.ImageItem(border='w')
        self.plotItem_p0.addItem(self.heatmap_p0)
        self.heatmap_p1_fg = pg.ImageItem(border='w')
        self.plotItem_p1_fg.addItem(self.heatmap_p1_fg)
        self.heatmap_p0_fg = pg.ImageItem(border='w')
        self.plotItem_p0_fg.addItem(self.heatmap_p0_fg)

        self.heatmap_p3 = pg.ImageItem(border='w')
        self.plotItem_p3.addItem(self.heatmap_p3)
        self.heatmap_p2 = pg.ImageItem(border='w')
        self.plotItem_p2.addItem(self.heatmap_p2)
        self.heatmap_p3_fg = pg.ImageItem(border='w')
        self.plotItem_p3_fg.addItem(self.heatmap_p3_fg)
        self.heatmap_p2_fg = pg.ImageItem(border='w')
        self.plotItem_p2_fg.addItem(self.heatmap_p2_fg)   

        # COLOR SCALE(LUT)
        colormap = pg.colormap.getFromMatplotlib('viridis')
        colormap_fg = pg.colormap.getFromMatplotlib('coolwarm')

        self.heatmap_p1.setLookupTable(colormap.getLookupTable())
        self.heatmap_p0.setLookupTable(colormap.getLookupTable())
        self.heatmap_p3.setLookupTable(colormap.getLookupTable())
        self.heatmap_p2.setLookupTable(colormap.getLookupTable())

        self.heatmap_p1_fg.setLookupTable(colormap_fg.getLookupTable())
        self.heatmap_p0_fg.setLookupTable(colormap_fg.getLookupTable())
        self.heatmap_p3_fg.setLookupTable(colormap_fg.getLookupTable())
        self.heatmap_p2_fg.setLookupTable(colormap_fg.getLookupTable())

        self.absScaleMin = 0
        self.absScaleMax = 1
        self.absScaleMin_fg = -1.0
        self.absScaleMax_fg = 1.0

        # SETTING LAYOUT
        self.BottomLayout = QHBoxLayout()
        self.radioLayout = QVBoxLayout() # 라디오 버튼 그룹을 위한 세로 라벨 레이아웃
        self.labelGroupLayout = QVBoxLayout() # 클래스 라벨 그룹을 위한 세로 라벨 레이아웃
        
        ## RADIO BUTTON GROUP
        self.radioGroupBox = QGroupBox("Port Type")
        self.BottomLayout.addWidget(self.radioGroupBox)
        self.radioButton0 = QRadioButton("Raw Mode")
        self.radioButton1 = QRadioButton("Butterworth Mode")
        self.radioButton1.setChecked(True)  # 기본 선택값 설정
        self.radioLayout.addWidget(self.radioButton0)
        self.radioLayout.addWidget(self.radioButton1)
        self.radioGroupBox.setLayout(self.radioLayout)

        ## File Num Label
        self.labelGroupBox = QGroupBox("Labels")
        self.labelGroupLayout = QVBoxLayout() # 그룹박스 내부 레이아웃

        self.label1 = QLabel("")
        font = QFont("Arial", 30)
        self.label1.setFont(font)
        self.label1.setAlignment(Qt.AlignCenter)
        self.labelGroupLayout.addWidget(self.label1)

        ## LABEL GROUP
        self.label2 = QLabel("Occ")
        self.label3 = QLabel("Loc")
        self.label4 = QLabel("Act")
        self.label2.setFont(font)
        self.label3.setFont(font)
        self.label4.setFont(font)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label3.setAlignment(Qt.AlignCenter)
        self.label4.setAlignment(Qt.AlignCenter)
        self.labelGroupLayout.addWidget(self.label2)
        self.labelGroupLayout.addWidget(self.label3)
        self.labelGroupLayout.addWidget(self.label4)
        self.labelGroupBox.setLayout(self.labelGroupLayout)
        self.BottomLayout.addWidget(self.labelGroupBox)

        ## BUTTON
        self.pushButton = QPushButton("정지 상태")
        self.pushButton.setStyleSheet("background-color: gray; color: black;")
        self.pushButton.setMaximumHeight(400)
        self.pushButton.setMinimumHeight(100)        
        self.pushButton.clicked.connect(self.toggleButtonState)
        self.BottomLayout.addWidget(self.pushButton)
        self.isButtonStopped = False
        self.layout.addLayout(self.BottomLayout)

        # QTimer
        self.timer = QTimer()
        self.timer.setInterval(0.05) # update per 0.1s
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(0) # 0 >> 100
    
    def update_graph(self):
        """시각화 업데이트"""
        for port_num in range(4):
            with csi_raw_array_shared[port_num].get_lock():  # 배열 잠금
                data_for_vis_raw = np.array(csi_raw_array_shared[port_num]).reshape(180, 192)  # 공유 배열에서 데이터 읽기
            with csi_bt_array_shared[port_num].get_lock():  # 배열 잠금
                data_for_vis_bt = np.array(csi_bt_array_shared[port_num]).reshape(180, 114)  # 공유 배열에서 데이터 읽기
            with csi_fg_array_shared[port_num].get_lock():
                data_for_vis_fg = np.array(csi_fg_array_shared[port_num]).reshape(180, 114)  # 공유 배열에서 데이터 읽기

            # 데이터를 시각화에 반영
            if port_num == 0:
                self.heatmap_p0_fg.setImage(data_for_vis_fg, levels=(self.absScaleMin_fg, self.absScaleMax_fg))
                if self.radioButton1.isChecked():
                    self.heatmap_p0.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p0.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
            elif port_num == 1:
                self.heatmap_p1_fg.setImage(data_for_vis_fg, levels=(self.absScaleMin_fg, self.absScaleMax_fg))
                if self.radioButton1.isChecked():
                    self.heatmap_p1.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p1.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
            elif port_num == 2:
                self.heatmap_p2_fg.setImage(data_for_vis_fg, levels=(self.absScaleMin_fg, self.absScaleMax_fg))
                if self.radioButton1.isChecked():
                    self.heatmap_p2.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p2.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
            elif port_num == 3:
                self.heatmap_p3_fg.setImage(data_for_vis_fg, levels=(self.absScaleMin_fg, self.absScaleMax_fg))
                if self.radioButton1.isChecked():
                    self.heatmap_p3.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p3.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
     
        self.label1.setText(str(self.labels_dict.get('time', '')))
        self.label2.setText(str(self.labels_dict.get('occ', '')))
        self.label3.setText(str(self.labels_dict.get('loc', '')))
        self.label4.setText(str(self.labels_dict.get('act', '')))
    
    def toggleButtonState(self):
        """버튼 상태 토글"""
        if self.isButtonStopped:
            self.isPushedBtn.value = False
            self.pushButton.setText("정지 상태")
            self.pushButton.setStyleSheet("background-color: gray; color: white;")
        else:
            self.isPushedBtn.value = True
            print("[⏰ ] 데이터 취득을 시작합니다. \n")
            self.pushButton.setText("취득중 상태")
            self.pushButton.setStyleSheet("background-color: blue; color: black;")
        self.isButtonStopped = not self.isButtonStopped

    def closeEvent(self, event):
        """창 닫기 이벤트"""
        terminate_process()  # 모든 프로세스 종료
        event.accept()

# ======================================================= #
#                   메인 함수                              #
# ======================================================= #
def main():
    # 명령행 인수 파싱
    acq_bool, csi_dir, model_type, inf_sec, prev_sec = parse_argument()
    
    # 포트 설정
    port_names = ['MQTT0']
    
    # 프로세스 리스트
    global PROCESSES
    
    try:

        # MQTT Data Acquisition
        process = Process(
            target= data_mqttsub_process,
            args=(data_queues[0], garbage_counters[0], exit_flag)
        )
        process.daemon= True
        process.start()
        PROCESSES.append(process)


        # MQTT Data Processing
        process = Process(
            target=data_mqtt_processing_process,
            args=(data_queues[0], inference_queue, storage_queue, visualization_queues[0], exit_flag)    
        )
        process.daemon = True
        process.start()
        PROCESSES.append(process)



        # 신경망 추론 프로세스 시작
        inference_process = Process(
            target=neural_network_inference_process,
            args=(inference_queue, storage_queue, LABELS, exit_flag, acq_bool)
        )
        inference_process.daemon = True
        inference_process.start()
        PROCESSES.append(inference_process)
        
        # GUI 시작
        app = QApplication(sys.argv)
        window = CSIDataGraphicalWindow(LABELS, isPushedBtn)
        window.show()
        sys.exit(app.exec_())
        
    except KeyboardInterrupt:
        print("\n[❌] 프로그램 종료 중...")
    finally:
        # 종료 플래그 설정
        exit_flag.value = True
        
        # 모든 프로세스 종료 대기
        for process in PROCESSES:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
        
        print("[✅] 모든 프로세스가 종료되었습니다.")

if __name__ == "__main__":
    main()
