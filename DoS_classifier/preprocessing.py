import pyshark
import pandas as pd
import ipaddress

def preprocess_pcapng(file_path):
    cap = pyshark.FileCapture(file_path)
    data = []
    
    for packet in cap:
        try:
            if 'IP' in packet:
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                protocol = packet.transport_layer
                length = int(packet.length)
                src_port = packet[packet.transport_layer].srcport
                dst_port = packet[packet.transport_layer].dstport
                timestamp = float(packet.sniff_timestamp)
                
                # TCP 플래그 추가
                if protocol == 'TCP':
                    flags = int(packet.tcp.flags, 16)
                else:
                    flags = None
                
                # DoS 공격 유형 라벨링 (예: SYN Flood, UDP Flood, etc.)
                if protocol == 'TCP' and flags == '0x0002':  # SYN 플래그
                    label = 'SYN Flood'
                elif protocol == 'UDP':
                    label = 'UDP Flood'
                elif protocol == 'ICMP':
                    label = 'ICMP Flood'
                elif protocol == 'HTTP':
                    label = 'HTTP Flood'
                else:
                    label = 'Other'
                
                data.append([src_ip, dst_ip, protocol, length, src_port, dst_port, timestamp, flags, label])

        except AttributeError:
            continue

    columns = ['Source', 'Destination', 'Protocol', 'Length', 'Src_Port', 'Dst_Port', 'Timestamp', 'TCP_Flags', 'Label']
    df = pd.DataFrame(data, columns=columns)
    df['Source_IP'] = df['Source'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    df['Destination_IP'] = df['Destination'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    df['Protocol'] = df['Protocol'].map({'TCP': 1, 'UDP': 2, 'ICMP': 3, 'HTTP': 4})
    df.to_csv('Slowloris1 packets.csv', index=False)    #수정하기

# 예제 사용
file_path = 'D:/대학교/3학년/1학기/네트워크 보안/실습/slowloris1.pcapng'    #수정하기
preprocess_pcapng(file_path)
