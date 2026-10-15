#include "control_gate.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace go1_bridge;
void expect(bool value) { if (!value) throw std::runtime_error("control gate check failed"); }
State state() { State s; s.valid=true;s.received_s=1;s.battery=24;s.temperature=40;
    for(int i=0;i<12;++i)s.q[i]=i%3==2 ? -1.5 : (i%3==1 ? .8 : 0);
    return s; }
Command command() { Command c;c.q=state().q;c.sequence=1;c.timestamp_us=1000000;c.kp.fill(20);c.kd.fill(.5);return c; }
Gate ready() { Gate g;g.armed=true;expect(g.accept(command(),1000000,1)==Fault::None);expect(g.evaluate(state(),1)==Fault::None);return g; }
int main() {
    expect(ready().evaluate(state(),1.02)==Fault::None);
    auto g=ready();expect(g.evaluate(state(),1.2)==Fault::StaleState);
    auto s=state();s.received_s=1.2;g=ready();expect(g.evaluate(s,1.2)==Fault::StaleCommand);
    g=ready();auto c=command();expect(g.accept(c,1000000,1)==Fault::OutOfOrder);
    g=ready();c=command();c.sequence=2;c.q[0]=std::numeric_limits<double>::quiet_NaN();expect(g.accept(c,1000000,1)==Fault::InvalidCommand);
    g=ready();c=command();c.sequence=2;c.timestamp_us=1020001;expect(g.accept(c,1000000,1)==Fault::StaleCommand);
    g=ready();s=state();s.emergency_stop=true;expect(g.evaluate(s,1)==Fault::EmergencyStop);
    s.emergency_stop=false;expect(g.evaluate(s,1)==Fault::EmergencyStop); // latched
    g=ready();s=state();s.q[2]-=.451;expect(g.evaluate(s,1)==Fault::Tracking);
    g=ready();c=command();c.sequence=2;c.kp.fill(40);expect(g.accept(c,1000000,1)==Fault::None);
    s=state();s.q[0]=-.44;s.dq[0]=-6;expect(g.evaluate(s,1)==Fault::Effort);
    g=ready();s=state();s.dq[0]=6.01;expect(g.evaluate(s,1)==Fault::Velocity);
    g=ready();s=state();s.battery=20.9;expect(g.evaluate(s,1)==Fault::Battery);
    g=ready();s=state();s.temperature=61;expect(g.evaluate(s,1)==Fault::Temperature);
    g=ready();s=state();s.roll=.46;expect(g.evaluate(s,1)==Fault::Posture);
    g=ready();s=state();s.remote_centered=false;expect(g.evaluate(s,1)==Fault::RemoteNotCentered);
    g=ready();c=command();c.sequence=2;c.q[2]=-.8;expect(g.accept(c,1000000,1)==Fault::JointLimit);
    g=ready();c=command();c.sequence=2;c.q[0]=.06;expect(g.accept(c,1020000,1.02)==Fault::Slew);
    g=ready();c=command();c.sequence=2;expect(g.accept(c,1200000,1.2)==Fault::StaleCommand);
    g=Gate();g.armed=true;c=command();c.q[0]=.11;
    expect(g.accept(c,1000000,1)==Fault::None);expect(g.evaluate(state(),1)==Fault::Tracking);
    g=ready();s=state();s.q[0]=std::numeric_limits<double>::quiet_NaN();
    expect(g.evaluate(s,1)==Fault::NoState);expect(g.evaluate(state(),1)==Fault::NoState);
    g=ready();c=command();c.sequence=2;c.kp[0]=41;expect(g.accept(c,1000000,1)==Fault::InvalidCommand);
    g=ready();c=command();c.sequence=2;c.tau[0]=.1;expect(g.accept(c,1000000,1)==Fault::InvalidCommand);
    std::cout << "control gate checks passed\n";
}
