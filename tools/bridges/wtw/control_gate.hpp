#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace go1_bridge {
using Joints = std::array<double, 12>;
enum class Fault : int { None=0, NotArmed=1, NoState=2, StaleState=3, NoCommand=4,
    StaleCommand=5, InvalidCommand=6, JointLimit=7, Tracking=8, Effort=9,
    Posture=10, Battery=11, Temperature=12, Velocity=13, EmergencyStop=14,
    OutOfOrder=15, RemoteNotCentered=16, Slew=17, MotorMode=18 };
struct State {
    Joints q{}, dq{};
    double received_s=0, roll=0, pitch=0, battery=0, temperature=0;
    bool valid=false, emergency_stop=false, remote_centered=true;
};
struct Command {
    Joints q{}, dq{}, kp{}, kd{}, tau{};
    int64_t sequence=-1, timestamp_us=0;
    double received_s=0;
};
inline double lower(int i) { return i%3==0 ? -1.047 : (i%3==1 ? -.663 : -2.721); }
inline double upper(int i) { return i%3==0 ? 1.047 : (i%3==1 ? 2.966 : -.837); }

class Gate {
 public:
    bool armed=false, have_command=false;
    Fault latched=Fault::None;
    Command command;
    bool first_target_pending=false;
    Fault reject(Fault fault) { if(latched==Fault::None)latched=fault; return latched; }
    Fault accept(const Command& proposed, int64_t wall_us, double now) {
        if (latched!=Fault::None) return latched;
        if (!armed) return Fault::NotArmed;
        if (proposed.sequence<0) return reject(Fault::EmergencyStop);
        if (have_command && proposed.sequence<=command.sequence) return reject(Fault::OutOfOrder);
        if (have_command && now-command.received_s>.1) return reject(Fault::StaleCommand);
        if (proposed.timestamp_us>wall_us+20000 || wall_us-proposed.timestamp_us>100000)
            return reject(Fault::StaleCommand);
        for (int i=0;i<12;++i) {
            const double values[]={proposed.q[i],proposed.dq[i],proposed.kp[i],proposed.kd[i],proposed.tau[i]};
            for (double v:values) if (!std::isfinite(v)) return reject(Fault::InvalidCommand);
            if (proposed.kp[i]<0 || proposed.kp[i]>40 || proposed.kd[i]<0 || proposed.kd[i]>2 ||
                proposed.dq[i]!=0 || proposed.tau[i]!=0) return reject(Fault::InvalidCommand);
            if (proposed.q[i]<lower(i) || proposed.q[i]>upper(i)) return reject(Fault::JointLimit);
            if (have_command && std::abs(proposed.q[i]-command.q[i])>2.5*std::max(0.,now-command.received_s)+.002)
                return reject(Fault::Slew);
        }
        first_target_pending=!have_command;
        command=proposed; command.received_s=now; have_command=true;
        return Fault::None;
    }
    Fault evaluate(const State& state, double now) {
        if (latched!=Fault::None) return latched;
        if (state.emergency_stop) return reject(Fault::EmergencyStop);
        if (!armed) return Fault::NotArmed;
        if (!state.valid) return have_command ? reject(Fault::NoState) : Fault::NoState;
        if (!std::isfinite(state.received_s) || now-state.received_s>.1 || now<state.received_s)
            return reject(Fault::StaleState);
        if (!std::isfinite(state.roll) || !std::isfinite(state.pitch) ||
            std::abs(state.roll)>.45 || std::abs(state.pitch)>.45) return reject(Fault::Posture);
        if (!std::isfinite(state.battery) || state.battery<21) return reject(Fault::Battery);
        if (!std::isfinite(state.temperature) || state.temperature>60) return reject(Fault::Temperature);
        if (!state.remote_centered) return reject(Fault::RemoteNotCentered);
        if (!have_command) return Fault::NoCommand;
        if (now-command.received_s>.1 || now<command.received_s) return reject(Fault::StaleCommand);
        for (int i=0;i<12;++i) {
            if (!std::isfinite(state.q[i]) || !std::isfinite(state.dq[i])) return reject(Fault::NoState);
            if(state.q[i]<lower(i)-.15 || state.q[i]>upper(i)+.15) return reject(Fault::JointLimit);
            if (std::abs(state.dq[i])>6) return reject(Fault::Velocity);
            if(first_target_pending && std::abs(command.q[i]-state.q[i])>.1) return reject(Fault::Tracking);
            if (std::abs(command.q[i]-state.q[i])>.45) return reject(Fault::Tracking);
            const double effort=command.kp[i]*(command.q[i]-state.q[i])-command.kd[i]*state.dq[i];
            if (!std::isfinite(effort) || std::abs(effort)>20) return reject(Fault::Effort);
        }
        first_target_pending=false;
        return Fault::None;
    }
};
} // namespace go1_bridge
