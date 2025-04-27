### **1. Push Docker Images to AWS Elastic Container Registry (ECR)**
ECS Fargate pulls images from **ECR**.

#### **A. Create ECR Repositories**
```bash
aws ecr create-repository --repository-name backend
aws ecr create-repository --repository-name frontend
```

#### **B. Authenticate Docker to AWS ECR**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

#### **C. Tag & Push Docker Images**
```bash
docker build -t backend -f Dockerfile.backend .
docker build -t frontend -f Dockerfile.frontend .

docker tag backend <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/backend:latest
docker tag frontend <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/frontend:latest

docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/backend:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/frontend:latest
```

---

### **2. Create an ECS Cluster**
```bash
aws ecs create-cluster --cluster-name my-app-cluster
```

---

### **3. Define Fargate Task Definitions**
Create JSON files for backend & frontend **task definitions**.

#### **A. Backend Task Definition (backend-task.json)**
```json
{
  "family": "backend",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/backend:latest",
      "memory": 512,
      "cpu": 256,
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000
        }
      ]
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/ecsTaskExecutionRole"
}
```

#### **B. Frontend Task Definition (frontend-task.json)**
```json
{
  "family": "frontend",
  "containerDefinitions": [
    {
      "name": "frontend",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/frontend:latest",
      "memory": 512,
      "cpu": 256,
      "portMappings": [
        {
          "containerPort": 80,
          "hostPort": 80
        }
      ]
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/ecsTaskExecutionRole"
}
```

#### **C. Register Task Definitions**
```bash
aws ecs register-task-definition --cli-input-json file://backend-task.json
aws ecs register-task-definition --cli-input-json file://frontend-task.json
```

---

### **4. Create an ECS Fargate Service**
#### **A. Create a VPC, Subnets & Security Group**
```bash
aws ec2 create-vpc --cidr-block 10.0.0.0/16
aws ec2 create-subnet --vpc-id <VPC_ID> --cidr-block 10.0.1.0/24
aws ec2 create-subnet --vpc-id <VPC_ID> --cidr-block 10.0.2.0/24
aws ec2 create-security-group --group-name ecs-security --description "ECS Fargate Security Group"

aws ec2 authorize-security-group-ingress --group-id <SG_ID> --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id <SG_ID> --protocol tcp --port 8000 --cidr 0.0.0.0/0

```
Allow traffic on ports **80 & 8000**.

#### **B. Create Backend & Frontend Services**
```bash
aws ecs create-service --cluster my-app-cluster --service-name backend-service \
--task-definition backend --desired-count 1 --launch-type FARGATE \
--network-configuration "awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}"
```

```bash
aws ecs create-service --cluster my-app-cluster --service-name frontend-service \
--task-definition frontend --desired-count 1 --launch-type FARGATE \
--network-configuration "awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}"
```

---

### **5. Test Deployment**
Run:
```bash
aws ecs list-services --cluster my-app-cluster
```
Then, check **public IPs** via:
```bash
aws ec2 describe-network-interfaces
```
Now, visit:
- **Frontend:** `http://<PUBLIC_IP>`
- **Backend API:** `http://<PUBLIC_IP>:8000`

To **disable** ECS service, you can either **stop all running tasks** or **delete the service completely**.

### Stop the ECS Service (Temporarily Disable)
This stops the service but keeps its configuration intact:
```bash
aws ecs update-service --cluster my-app-cluster --service backend-service --desired-count 0
```
Replace `backend-service` with actual **service name**.

---

### Delete the ECS Service (Permanent Removal)
If you want to **permanently remove** the service, run:
```bash
aws ecs delete-service --cluster my-app-cluster --service backend-service --force
```
Using `--force` ensures AWS removes **all associated tasks** immediately.

---

### Delete the Entire ECS Cluster
If you want to **remove everything**, run:
```bash
aws ecs delete-cluster --cluster my-app-cluster
```
