from scapy.all import rdpcap, IP,  ICMP, TCP, UDP, ARP, STP, DNS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read the PCAP file
pcap_file = 'capture.pcapng'
packets = rdpcap(pcap_file)

protocol_counts = {}
packet_lengths = []
packet_times = [packet.time for packet in packets]  # Get packet timestamps

# Analyze the packets
for packet in packets:
    if ARP in packet:
        protocol = "ARP"
    elif ICMP in packet:
        protocol = "ICMP"
    elif TCP in packet:
        if bytes(packet[TCP].payload):
            payload = bytes(packet[TCP].payload)
            if b'HTTP' in payload:  # Check if HTTP is in the TCP payload
                protocol = "HTTP"
            elif DNS in packet:
                protocol = "DNS"
            else:
                protocol = "TCP"
        else:
            protocol = "TCP"
    elif UDP in packet:
        protocol = "UDP"
    elif STP in packet:
        protocol = "STP"
    elif IP in packet:
        protocol = "IP"
    else:
        protocol = "Other"

    if protocol in protocol_counts:
        protocol_counts[protocol] += 1
    else:
        protocol_counts[protocol] = 1
    
    packet_lengths.append(len(packet))

# Create DataFrame
df = pd.DataFrame({'time': packet_times, 'length': packet_lengths})

# Convert time to numeric, coercing any errors to NaN
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Drop rows with NaN values in 'time' column
df.dropna(subset=['time'], inplace=True)

# Traffic time series analysis for anomaly detection
df['time_bucket'] = pd.cut(df['time'], bins=np.arange(min(df['time']), max(df['time']), 1))
time_series = df.groupby('time_bucket')['length'].count()

# Use subplots to display all charts in a single figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout(pad=5.0)

# 1. Protocol Distribution Bar Chart
axs[0, 0].bar(protocol_counts.keys(), protocol_counts.values(), color=plt.cm.Paired.colors)
axs[0, 0].set_xlabel('Protocol')
axs[0, 0].set_ylabel('Count')
axs[0, 0].set_title('Protocol Distribution (Bar Chart)')

# 2. Packet Length Over Time
axs[0, 1].plot(packet_times, packet_lengths, marker='o', linestyle='-', color='blue')
axs[0, 1].set_xlabel('Time (seconds)')
axs[0, 1].set_ylabel('Packet Length (bytes)')
axs[0, 1].set_title('Packet Length Over Time')

# 3. Time Series and Anomaly Detection
axs[1, 0].plot(time_series.index.astype(str), time_series.values, marker='o', linestyle='-', color='red')
axs[1, 0].set_xlabel('Time (1-second intervals)')
axs[1, 0].set_ylabel('Packet Count')
axs[1, 0].set_title('Traffic Time Series (Anomaly Detection)')

# 4. Show the Pairwise Correlation Matrix
correlation_df = pd.DataFrame({
    'length': packet_lengths,
    'time': packet_times
})
sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', center=0, ax=axs[1, 1])
axs[1, 1].set_title('Pairwise Correlation Matrix')

# Show the plots
plt.show()
