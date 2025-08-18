#include <iostream>
#include "FSM/State_Rl.h"
#include <cmath>
#define _USE_MATH_DEFINES
using namespace std;


State_Rl::State_Rl(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::RL, "rl"),
    _est(ctrlComp->estimator)
{

}

void State_Rl::enter()//初始化
{
    // load policy
    model_path = "model_mixed_him_baseline_kp30.pt";
    load_policy();

    // initialize record
    action_buf = torch::zeros({history_length,12},device);
//    obs_buf = torch::zeros({history_length,30},device);
    obs_buf = torch::zeros({history_length,45},device);
    last_action = torch::zeros({1,12},device);

    action_buf.to(torch::kHalf);
    obs_buf.to(torch::kHalf);
    last_action.to(torch::kHalf);

    // initialize default values
    gravity(0,0) = 0.0;
    gravity(1,0) = 0.0;
    gravity(2,0) = -1.0;

    _gxFilter = new LPFilter(infer_dt,3.0);
    _gyFilter = new LPFilter(infer_dt,3.0);
    _gzFilter = new LPFilter(infer_dt,3.0);

    forward(0,0) = 1.0;
    forward(1,0) = 0.0;
    forward(2,0) = 0.0;

    for (int j = 0; j < 12; j++)
    {
        action_temp.push_back(0.0);
        //action.push_back(_lowState->motorState[j].q);
        //prev_action.push_back(_lowState->motorState[j].q);
	    action.push_back(init_pos[j]);
        prev_action.push_back(init_pos[j]);

    }
//    for (int j = 0; j < 12; j++)
//    {
//        action.push_back(0.0);
//    }

    for (int i = 0; i < history_length; i++)
    {
        torch::Tensor obs_tensor = get_obs();
        // append obs to obs buffer
        obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);
    }
    std::cout << "init finised predict" << std::endl;

    for (int i = 0; i < 200; i++)
    {
        model_infer();
    }

    // initialize thread
    threadRunning = true;
    infer_thread = new std::thread(&State_Rl::infer,this);


    // smooth transition of kp and kd
    for (int j = 0; j < 12; j++)
    {
        stand_kd[j] = _lowCmd->motorCmd[j].Kd;
        stand_kp[j] = _lowCmd->motorCmd[j].Kp;
    }

    for (int j = 0; j < 12; j++)
    {
      action_filters.push_back(new LPFilter(0.002,20.0));
    }
}

void State_Rl::run()
{   
    if (_percent_1 < 1)
    {
        _percent_1 += (float) 1 / _duration_1;
        _percent_1 = _percent_1 > 1 ? 1 : _percent_1;
        if (_percent_1 < 1) {
            for (int j = 0; j < 12; j++) {
                _lowCmd->motorCmd[j].mode = 10;
		_lowCmd->motorCmd[j].q = (1 - _percent_1) * _startPos[j] + _percent_1 * init_pos[j];
                _lowCmd->motorCmd[j].dq = 0;
                _lowCmd->motorCmd[j].Kp = (1 - _percent_1) * stand_kp[j] + _percent_1 * Kp;
                _lowCmd->motorCmd[j].Kd = (1 - _percent_1) * stand_kd[j] + _percent_1 * Kd;
                _lowCmd->motorCmd[j].tau = 0;
            }
        } 
    }
    else
    {
        write_cmd_lock.lock();
        for (int j = 0; j < 12; j++)
        {
            _lowCmd->motorCmd[j].mode = 10;
            _lowCmd->motorCmd[j].q = action[j];//action_filters[j]->getValue();
            _lowCmd->motorCmd[j].dq = 0;
            _lowCmd->motorCmd[j].Kp = Kp;
            _lowCmd->motorCmd[j].Kd = Kd;
            _lowCmd->motorCmd[j].tau = 0;

        }
        write_cmd_lock.unlock();
    }
   
}

void State_Rl::exit()
{
    _percent_1 = 0;
    _percent_2 = 0;
    threadRunning = false;
    infer_thread->join();
}

torch::Tensor State_Rl::get_obs()//获取观测
{
    std::vector<float> obs;
    // compute gravity
    _B2G_RotMat = _lowState->getRotMat();
    _G2B_RotMat = _B2G_RotMat.transpose();

    Vec3 angvel = _lowState->getGyro();
    Vec3 projected_gravity = _G2B_RotMat*gravity;
    Vec3 projected_forward = _G2B_RotMat*forward;//unuse

    // obs_buf =torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
    //                 #self.projected_gravity,
    //                 self.base_euler_xyz * self.obs_scales.quat,
    //                 self.commands[:, :3] * self.commands_scale,#xy+航向角速度
    //                 self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
    //                 self.reindex(self.dof_vel * self.obs_scales.dof_vel),
    //                 self.action_history_buf[:,-1]),dim=-1)

    // gravity
    _gxFilter->addValue(angvel(0,0));
    _gyFilter->addValue(angvel(1,0));
    _gzFilter->addValue(angvel(2,0));
//
    obs.push_back(_gxFilter->getValue()*0.25);//ang_vel = 0.25
    obs.push_back(_gyFilter->getValue()*0.25);
    obs.push_back(_gzFilter->getValue()*0.25);

    for (int i = 0; i < 3; ++i)
    {
        obs.push_back(projected_gravity(i,0));
    }

    // cmd
    rx = rx * (1 - smooth) + (std::fabs(_lowState->userValue.rx) < dead_zone ? 0.0 : _lowState->userValue.rx) * smooth;
    ly = ly * (1 - smooth) + (std::fabs(_lowState->userValue.ly) < dead_zone ? 0.0 : _lowState->userValue.ly) * smooth;
    lx = lx * (1 - smooth) + (std::fabs(_lowState->userValue.lx) < dead_zone ? 0.0 : _lowState->userValue.lx) * smooth;

    float max = 1.0;
    float min = -1.0;
//    float vel = std::max(std::min(_lowState->userValue.ly, max), min);
//    if(vel > 0)
//    {
//        vel = 1.0;
//    }
//    else if(vel < 0)
//    {
//        vel = -1.0;
//    }
//    float  vel = 0.6;
    float rot = rx*3.14;
    float vel = ly*2.0;
    float lat_vel = lx*2.0;

    double heading = atan2((double)forward(1,0), (double)forward(0,0));
    double angle = (double)rot - heading;
    angle = fmod(angle,2.0*M_PI);
    if(angle > M_PI)
    {
        angle = angle - 2.0*M_PI;
    }
    angle = angle*0.5;
    angle = std::max(std::min((float)angle, max), min);
    angle = angle * 0.25;

    obs.push_back(vel);//控制指令x
    obs.push_back(lat_vel);//控制指令y
    obs.push_back(angle);//控制指令yaw rate

    // pos q joint
    for (int i = 0; i < 12; ++i)
    {
        float pos = (_lowState->motorState[i].q  - init_pos[i])* pos_scale;
        obs.push_back(pos);
    }
    // vel q joint
    for (int i = 0; i < 12; ++i)
    {
        float vel = _lowState->motorState[i].dq * vel_scale;
        obs.push_back(vel);
    }

    // last action
    //float index[12] = {3,4,5,0,1,2,9,10,11,6,7,8};
    for (int i = 0; i < 12; ++i)
    {
        obs.push_back(action_temp[i]);//历史  self.cfg.env.history_len, self.num_dofs
    }

    // gravity,cmd,dof_pos,dof_vel to tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
//    torch::Tensor obs_tensor = torch::from_blob(obs.data(),{1,30},options).to(device);
    torch::Tensor obs_tensor = torch::from_blob(obs.data(),{1,45},options).to(device);

    return obs_tensor;
}

torch::Tensor State_Rl::model_infer()//接口doghome
{   
    torch::NoGradGuard no_grad;

    torch::Tensor obs_tensor = get_obs();//原始观测
    //obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);
//    auto obs_buf_batch = obs_buf.unsqueeze(0);
//    auto action_buf_batch = action_buf.unsqueeze(0);
//
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(obs_buf_batch);
//    inputs.push_back(action_buf_batch);

    //auto obs_batch = obs_tensor.unsqueeze(0);
    auto obs_buf_batch = obs_buf.unsqueeze(0);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(obs_tensor.to(torch::kHalf));
    inputs.push_back(obs_buf_batch.to(torch::kHalf));

    // Execute the model and turn its output into a tensor.
    torch::Tensor action_tensor = model.forward(inputs).toTensor();
    action_buf = torch::cat({action_buf.index({Slice(1,None),Slice()}),action_tensor},0);

    torch::Tensor action_blend_tensor = 0.8*action_tensor + 0.2*last_action;
    last_action = action_tensor.clone();

    obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);

    return action_blend_tensor;
}

void State_Rl::infer()
{
    while(threadRunning)
    {
        long long _start_time = getSystemTime();

        //torch::Tensor obs_tensor = get_obs();
        // append obs to obs buffer
        //obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);

        torch::Tensor action_raw = model_infer();
//        obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);
        // action filter
//        action_raw = 0.8*action_raw + 0.2*last_action;
        //last_action = action_raw.clone();
        // append to action buffer
        // action_buf = torch::cat({action_buf.index({Slice(1,None),Slice()}),action_raw},0);
        // assign to control
        action_raw = action_raw.squeeze(0);
        // move to cpu
	    action_raw = action_raw.to(torch::kFloat32);
        action_raw = action_raw.to(torch::kCPU);
        // assess the result
        auto action_getter = action_raw.accessor<float,1>();

        write_cmd_lock.lock();
        for (int j = 0; j < 12; j++)
        {
//            float  action_value = std::max(std::min(action_getter[j]* action_scale[j], action_delta_max), action_delta_min);
//            action.at(j) = action_value + init_pos[j];
//            action_temp.at(j) = action_value/action_scale[j];
            action[j] = action_getter[j] * action_scale[j] + init_pos[j];
            action_temp[j] = action_getter[j];//原始值
        }
        write_cmd_lock.unlock();

        absoluteWait(_start_time, (long long)(infer_dt * 1000000));
    }
    threadRunning = false;

}

void State_Rl::load_policy()
{   
    std::cout << model_path << std::endl;
    // load model from check point
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    device= torch::kCPU;
    if (torch::cuda::is_available()){
        device = torch::kCUDA;
    }
    model = torch::jit::load(model_path);
    std::cout << "load model is successed!" << std::endl;
    model.to(device);
    model.to(torch::kHalf);
    std::cout << "load model to device!" << std::endl;
    model.eval();
}

FSMStateName State_Rl::checkChange()
{
    if (_lowState->userCmd == UserCommand::L2_B)
    {
        return FSMStateName::PASSIVE;
    }
    if(_lowState->userCmd == UserCommand::L2_X){
        return FSMStateName::FIXEDSTAND;
    }
    else{
        return FSMStateName::RL;
    }
}
