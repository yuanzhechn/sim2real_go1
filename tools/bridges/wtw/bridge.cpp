// Independent adapter for WTW's public LCM protocol. Upstream files stay in the sibling clone.
#include "control_gate.hpp"
#include "pd_tau_targets_lcmt.hpp"
#include "leg_control_data_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "go1_bridge_status_lcmt.hpp"
#include <lcm/lcm-cpp.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef HAVE_UNITREE_SDK
#include <unitree_legged_sdk/unitree_legged_sdk.h>
#include <unitree_legged_sdk/joystick.h>
#endif
using Clock=std::chrono::steady_clock;
using namespace go1_bridge;
static_assert(int(Fault::StaleCommand)==go1_bridge_status_lcmt::FAULT_STALE_COMMAND,"LCM fault contract mismatch");
static_assert(int(Fault::NoCommand)==go1_bridge_status_lcmt::FAULT_NO_COMMAND,"LCM idle contract mismatch");
static_assert(int(Fault::None)==go1_bridge_status_lcmt::FAULT_NONE,"LCM normal contract mismatch");
volatile std::sig_atomic_t interrupted=0;
void stop_handler(int) { interrupted=1; }
double steady_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
int64_t wall_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
struct Options {
    bool hardware=false,protected_robot=false,sport_stopped=false;
    double seconds=10;
    std::string url="udpm://239.255.76.67:7667?ttl=0",ip="192.168.123.10";
    int target_port=8007,local_port=8090;
};

class Bridge {
    Options options;
    lcm::LCM receiver,publisher;
    std::mutex mutex;
    std::mutex packet_mutex;
    Gate gate;
    State current;
    bool was_r2=false;
    std::atomic<bool> running{true};
    std::atomic<double> packet_updated_s{0};
    std::atomic<bool> sender_fault{false},hardware_armed{false};
    std::thread rx_thread,send_thread,command_thread;
    int64_t sequence=0,cycles=0,missed=0;
    double maximum_interval=0,previous_cycle=0,started=0;
    std::string prefix;
    leg_control_data_lcmt legs{};
    state_estimator_lcmt body{};
    rc_command_lcmt remote{};
#ifdef HAVE_UNITREE_SDK
    std::unique_ptr<UNITREE_LEGGED_SDK::UDP> udp;
    std::unique_ptr<UNITREE_LEGGED_SDK::Safety> sdk_safety;
    UNITREE_LEGGED_SDK::LowState sdk_state{};
    UNITREE_LEGGED_SDK::LowCmd sdk_command{};
    uint32_t last_tick=0;
#endif
    void periodic(const std::function<void()>& callback) {
        auto next=Clock::now();
        while(running) {
            callback(); next+=std::chrono::microseconds(2000);
            if (Clock::now()>next+std::chrono::milliseconds(20)) next=Clock::now();
            std::this_thread::sleep_until(next);
        }
    }
    void on_command(const lcm::ReceiveBuffer*,const std::string&,const pd_tau_targets_lcmt* message) {
        Command command;command.sequence=message->id;command.timestamp_us=message->timestamp_us;
        for(int i=0;i<12;++i) {
            command.q[i]=message->q_des[i];command.dq[i]=message->qd_des[i];
            command.kp[i]=message->kp[i];command.kd[i]=message->kd[i];command.tau[i]=message->tau_ff[i];
        }
        std::lock_guard<std::mutex> lock(mutex);
        gate.accept(command,wall_us(),steady_s());
    }
    void read_hardware(double now) {
#ifdef HAVE_UNITREE_SDK
        udp->GetRecv(sdk_state);
        double norm=0;for(float q:sdk_state.imu.quaternion)norm+=q*q;
        const bool valid_imu=std::isfinite(norm) && norm>.5 && norm<1.5;
        if(!valid_imu)current.valid=false;
        if (valid_imu && sdk_state.tick!=last_tick) {
            last_tick=sdk_state.tick;current.received_s=now;current.valid=true;
        }
        current.roll=sdk_state.imu.rpy[0];current.pitch=sdk_state.imu.rpy[1];
        current.battery=0;for(auto cell:sdk_state.bms.cell_vol)current.battery+=cell*.001;
        current.temperature=0;
        for(int i=0;i<12;++i) {
            current.q[i]=sdk_state.motorState[i].q;current.dq[i]=sdk_state.motorState[i].dq;
            current.temperature=std::max(current.temperature,double(sdk_state.motorState[i].temperature));
            if(sdk_state.motorState[i].mode!=0 && sdk_state.motorState[i].mode!=10)
                gate.reject(Fault::MotorMode);
            legs.tau_est[i]=sdk_state.motorState[i].tauEst;
        }
        xRockerBtnDataStruct buttons{};
        std::memcpy(&buttons,sdk_state.wirelessRemote.data(),40);
        current.emergency_stop=buttons.btn.components.B;
        current.remote_centered=std::abs(buttons.lx)<=.08 && std::abs(buttons.ly)<=.08 && std::abs(buttons.rx)<=.08;
        const bool r2=buttons.btn.components.R2;
        if(current.valid && r2 && !was_r2 && current.remote_centered && !current.emergency_stop)
            gate.armed=true;
        was_r2=r2;
        hardware_armed=gate.armed;
        remote.left_stick[0]=buttons.lx;remote.left_stick[1]=buttons.ly;
        remote.right_stick[0]=buttons.rx;remote.right_stick[1]=buttons.ry;
        remote.right_lower_right_switch=r2;
        remote.left_lower_left_switch=buttons.btn.components.L2;
        remote.left_upper_switch=buttons.btn.components.L1;remote.right_upper_switch=buttons.btn.components.R1;
        remote.mode=current.emergency_stop ? 1 : 0;
        for(int i=0;i<3;++i) { body.rpy[i]=sdk_state.imu.rpy[i];body.omegaBody[i]=sdk_state.imu.gyroscope[i];body.aBody[i]=sdk_state.imu.accelerometer[i]; }
        for(int i=0;i<4;++i) { body.quat[i]=sdk_state.imu.quaternion[i];body.contact_estimate[i]=sdk_state.footForce[i]; }
#else
        (void)now;
#endif
    }
    void make_hardware_command(Fault fault) {
#ifdef HAVE_UNITREE_SDK
        for(int i=0;i<12;++i) {
            auto& motor=sdk_command.motorCmd[i];
            motor.mode=gate.armed ? 10 : 0;
            motor.q=UNITREE_LEGGED_SDK::PosStopF;motor.dq=0;motor.Kp=0;motor.Kd=gate.armed ? 1 : 0;motor.tau=0;
            if(fault==Fault::None) {
                motor.q=gate.command.q[i];motor.Kp=gate.command.kp[i];motor.Kd=gate.command.kd[i];
            }
        }
        if(fault==Fault::None) {
            sdk_safety->PositionLimit(sdk_command);
            if(sdk_safety->PowerProtect(sdk_command,sdk_state,1)<0) {
                gate.reject(Fault::Effort);
                for(int i=0;i<12;++i) { sdk_command.motorCmd[i].q=UNITREE_LEGGED_SDK::PosStopF;sdk_command.motorCmd[i].Kp=0;sdk_command.motorCmd[i].Kd=1; }
            }
        }
        {
            std::lock_guard<std::mutex> lock(packet_mutex);
            if(sender_fault)for(int i=0;i<12;++i) {
                sdk_command.motorCmd[i].q=UNITREE_LEGGED_SDK::PosStopF;
                sdk_command.motorCmd[i].Kp=0;sdk_command.motorCmd[i].Kd=1;
            }
            udp->SetSend(sdk_command);packet_updated_s=steady_s();
        }
#else
        (void)fault;
#endif
    }
    void control() {
        const double now=steady_s();
        std::lock_guard<std::mutex> lock(mutex);
        if(previous_cycle) {
            double interval=now-previous_cycle;maximum_interval=std::max(maximum_interval,interval);
            if(interval>.003)++missed;
        }
        previous_cycle=now;++cycles;
        if(sender_fault)gate.reject(Fault::StaleCommand);
        if(options.hardware)read_hardware(now);
        else { current.received_s=now;current.valid=true; }
        Fault fault=gate.evaluate(current,now);
        if(options.hardware)make_hardware_command(fault);
        else if(fault==Fault::None) {
            for(int i=0;i<12;++i) {
                const double delta=std::max(-.004,std::min(.004,gate.command.q[i]-current.q[i]));
                current.q[i]+=delta;current.dq[i]=delta/.002;
            }
        } else current.dq.fill(0);
        // State is published at 100Hz; safety and hardware send/receive continue at 500Hz.
        if(cycles%5)return;
        const int64_t stamp=wall_us();
        legs.timestamp_us=body.timestamp_us=stamp;legs.id=body.id=++sequence;
        for(int i=0;i<12;++i) { legs.q[i]=current.q[i];legs.qd[i]=current.dq[i]; }
        go1_bridge_status_lcmt status{};
        status.timestamp_us=stamp;status.sequence=sequence;status.fault=static_cast<int>(fault);
        status.armed=gate.armed;status.simulated=!options.hardware;
        status.battery_voltage=current.battery;status.max_motor_temperature=current.temperature;
        status.command_age_ms=gate.have_command ? (now-gate.command.received_s)*1000 : -1;
        status.cycles=cycles;status.missed_deadlines=missed;
        for(int i=0;i<12;++i) { status.q[i]=current.q[i];status.dq[i]=current.dq[i]; }
        for(int i=0;i<3;++i) { status.rpy[i]=body.rpy[i];status.omega[i]=body.omegaBody[i]; }
        for(int i=0;i<4;++i) { status.quat[i]=body.quat[i];status.foot_force[i]=body.contact_estimate[i]; }
        status.state_age_ms=current.valid ? (now-current.received_s)*1000 : -1;
        status.emergency_stop=current.emergency_stop;status.r2=remote.right_lower_right_switch;
        publisher.publish(prefix+"leg_control_data",&legs);
        publisher.publish(prefix+"state_estimator_data",&body);
        publisher.publish(prefix+"rc_command",&remote);
        publisher.publish(prefix+"status",&status);
    }
 public:
    explicit Bridge(Options options):options(options),receiver(options.url),publisher(options.url),prefix(options.hardware?"GO1_WTW_":"GO1_WTW_TEST_") {
        if(!receiver.good() || !publisher.good())throw std::runtime_error("LCM unavailable");
        receiver.subscribe(prefix+"pd_plustau_targets",&Bridge::on_command,this)->setQueueCapacity(1);
        if(options.hardware) {
#ifdef HAVE_UNITREE_SDK
            udp.reset(new UNITREE_LEGGED_SDK::UDP(UNITREE_LEGGED_SDK::LOWLEVEL,options.local_port,options.ip.c_str(),options.target_port));
            sdk_safety.reset(new UNITREE_LEGGED_SDK::Safety(UNITREE_LEGGED_SDK::LeggedType::Go1));
            udp->InitCmdData(sdk_command);
            for(int i=0;i<12;++i) { sdk_command.motorCmd[i].mode=0;sdk_command.motorCmd[i].Kp=0;sdk_command.motorCmd[i].Kd=0;sdk_command.motorCmd[i].tau=0; }
            udp->SetSend(sdk_command);
            packet_updated_s=steady_s();
#else
            throw std::runtime_error("Compiled without SDK; hardware unavailable");
#endif
        } else {
            current.battery=24;current.temperature=35;gate.armed=true;
            const double q[]={-.1,.8,-1.5,.1,.8,-1.5,-.1,1.,-1.5,.1,1.,-1.5};
            std::copy(q,q+12,current.q.begin());body.quat[0]=1;
        }
    }
    void run() {
        started=steady_s();
        command_thread=std::thread([this]{while(running)receiver.handleTimeout(5);});
#ifdef HAVE_UNITREE_SDK
        if(options.hardware) {
            rx_thread=std::thread([this]{periodic([this]{udp->Recv();});});
            send_thread=std::thread([this]{periodic([this]{
                std::lock_guard<std::mutex> lock(packet_mutex);
                // Stop even if the control loop stops updating its validated packet.
                if(sender_fault || steady_s()-packet_updated_s.load()>.1) {
                    sender_fault=true;
                    UNITREE_LEGGED_SDK::LowCmd damping{};udp->InitCmdData(damping);
                    for(int i=0;i<12;++i) {
                        damping.motorCmd[i].mode=hardware_armed ? 10 : 0;
                        damping.motorCmd[i].q=UNITREE_LEGGED_SDK::PosStopF;
                        damping.motorCmd[i].Kp=0;damping.motorCmd[i].Kd=hardware_armed ? 1 : 0;damping.motorCmd[i].tau=0;
                    }
                    udp->SetSend(damping);
                }
                udp->Send();
            });});
        }
#endif
        auto next=Clock::now();
        while(!interrupted && steady_s()-started<options.seconds) {
            control();next+=std::chrono::microseconds(2000);
            if(Clock::now()>next+std::chrono::milliseconds(20))next=Clock::now();
            std::this_thread::sleep_until(next);
        }
        int final_fault;
        { std::lock_guard<std::mutex> lock(mutex);final_fault=int(gate.latched);gate.reject(Fault::EmergencyStop); }
        if(options.hardware)for(int i=0;i<10;++i) { control();std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
        running=false;
        command_thread.join();if(rx_thread.joinable())rx_thread.join();if(send_thread.joinable())send_thread.join();
        std::cout << "{\"simulated\":" << (!options.hardware?"true":"false") << ",\"cycles\":" << cycles
                  << ",\"elapsed_s\":" << steady_s()-started << ",\"intervals_over_3ms\":" << missed
                  << ",\"max_interval_ms\":" << maximum_interval*1000 << ",\"fault\":" << final_fault << "}\n";
    }
};
int main(int argc,char**argv) {
    try {
        Options options;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--enable-hardware")options.hardware=true;
            else if(arg=="--protected")options.protected_robot=true;
            else if(arg=="--sport-mode-stopped")options.sport_stopped=true;
            else if(arg=="--simulate")options.hardware=false;
            else if(arg=="--seconds" && i+1<argc)options.seconds=std::stod(argv[++i]);
            else if(arg=="--lcm-url" && i+1<argc)options.url=argv[++i];
            else if(arg=="--target-ip" && i+1<argc)options.ip=argv[++i];
            else throw std::runtime_error("Unknown/incomplete argument: "+arg);
        }
        if(!std::isfinite(options.seconds) || options.seconds<=0 || options.seconds>30)
            throw std::runtime_error("Duration must be in (0,30] seconds");
        if(options.hardware && (!options.protected_robot || !options.sport_stopped))
            throw std::runtime_error("Hardware requires --protected --sport-mode-stopped; R2 arms, B latches damping");
        std::signal(SIGINT,stop_handler);std::signal(SIGTERM,stop_handler);
        Bridge bridge(options);bridge.run();return 0;
    }catch(const std::exception& error){std::cerr << error.what()<<"\n";return 1;}
}
