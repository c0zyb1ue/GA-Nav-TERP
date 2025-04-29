import torch

# PTH 파일 경로
file_path = "/root/catkin_ws/src/terp/TERP/tmp/ddpg/actor_ddpg.zip"

# 모델 로딩 테스트
try:
    model_data = torch.load(file_path, map_location=torch.device('cpu'))  # CPU에서 로드
    
    # 모델 데이터 타입 확인
    model_type = type(model_data)
    
    # 모델 가중치 키 확인 (state_dict 방식인지 확인)
    model_keys = model_data.keys() if isinstance(model_data, dict) else None
    print(f"model: {model_data}")

    # 출력 결과 정리
    model_info = {
        "status": "success",
        "model_type": str(model_type),
        "keys": list(model_keys) if model_keys else "Not a dictionary"
    }

except Exception as e:
    model_info = {
        "status": "error",
        "error_message": str(e)
    }

print(model_info)
