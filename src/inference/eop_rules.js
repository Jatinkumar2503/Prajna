/* ============================================================
   PRAJNA DIGITAL IAEA EOP DECISION-SUPPORT ENGINE — eop_rules.js
   Compliant with IAEA Safety Standards Series & AERB/NPP/EOP Guidelines.
   Translates raw neural network logits and physical parameter excursions
   into ranked, prioritized, and time-bounded operator emergency actions.
   ============================================================ */

var PrajnaEOPRules = (function () {
  'use strict';

  var EOP_DATABASE = {
    0: {
      code: "EOP-00-NORM",
      title: "Nominal Base-Load Operation",
      titleHindi: "सामान्य आधार-भार संचालन",
      priority: "NORMAL",
      urgencyLevel: "MONITORING",
      timeToBreachSeconds: 9999,
      actions: [
        { id: 1, step: "1.0", title: "Continuous Telemetry Audit", desc: "Maintain secondary loop thermal-hydraulic balance & monitor core ΔT.", tier: "ROUTINE", timeLimitSec: 3600 },
        { id: 2, step: "1.1", title: "Verify Xenon Equilibrium", desc: "Ensure reactivity feedback remains within ±10 pcm/h target envelope.", tier: "ROUTINE", timeLimitSec: 7200 }
      ]
    },
    1: {
      code: "EOP-E1-LOCA",
      title: "Loss of Coolant Accident & ECCS Emergency Mitigation",
      titleHindi: "आपातकालीन कोर शीतलन प्रणाली तुरंत सक्रिय करें (LOCA)",
      priority: "CRITICAL",
      urgencyLevel: "IMMEDIATE TRIP & INJECTION",
      timeToBreachSeconds: 45,
      actions: [
        { id: 1, step: "1.0", title: "Verify Reactor SCRAM & Turbine Trip", desc: "Confirm all control rod bottom lights illuminated & turbine stop valves closed.", tier: "CRITICAL", timeLimitSec: 5 },
        { id: 2, step: "2.0", title: "Actuate High-Pressure Safety Injection (HPSI)", desc: "Verify emergency borated coolant injection valves open to replenish primary inventory.", tier: "CRITICAL", timeLimitSec: 15 },
        { id: 3, step: "3.0", title: "Isolate Reactor Containment Building", desc: "Seal purge isolation valves & start containment spray pumps to suppress pressure.", tier: "MANDATORY", timeLimitSec: 30 },
        { id: 4, step: "4.0", title: "Trip Primary Reactor Coolant Pumps (RCPs)", desc: "Prevent pump cavitation damage during severe depressurization & two-phase voiding.", tier: "HIGH", timeLimitSec: 45 }
      ]
    },
    2: {
      code: "EOP-E2-RIA",
      title: "Control Rod Ejection & Prompt-Criticality Mitigation",
      titleHindi: "स्वचालित रिएक्टर ट्रिप / स्क्रैम (SCRAM) एवं डोप्लर स्थिरीकरण",
      priority: "CRITICAL",
      urgencyLevel: "PROMPT TRIP REQUIRED",
      timeToBreachSeconds: 8,
      actions: [
        { id: 1, step: "1.0", title: "Manual Emergency SCRAM Actuation", desc: "Depress dual emergency trip push-buttons to drop all gravity shutoff rods.", tier: "CRITICAL", timeLimitSec: 2 },
        { id: 2, step: "2.0", title: "Inject High-Concentration Soluble Boron", desc: "Actuate Standby Liquid Control (SLC) fast boration valves to ensure subcriticality.", tier: "CRITICAL", timeLimitSec: 10 },
        { id: 3, step: "3.0", title: "Monitor Fuel Clad Thermal Strain", desc: "Verify peak clad temperature does not exceed 1204°C (10 CFR 50.46 limit).", tier: "MANDATORY", timeLimitSec: 30 },
        { id: 4, step: "4.0", title: "Establish Decay Heat Removal Sink", desc: "Align secondary atmospheric dump valves to maintain core thermal equilibrium.", tier: "HIGH", timeLimitSec: 60 }
      ]
    },
    3: {
      code: "EOP-E3-SGTR",
      title: "Steam Generator Tube Rupture Secondary Containment Isolation",
      titleHindi: "भाप जनरेटर पृथक्करण एवं रेडियोधर्मी रिसाव रोकथाम (SGTR)",
      priority: "HIGH",
      urgencyLevel: "PRIMARY DEPRESSURIZATION",
      timeToBreachSeconds: 90,
      actions: [
        { id: 1, step: "1.0", title: "Identify & Isolate Ruptured Steam Generator", desc: "Close Main Steam Isolation Valve (MSIV) and feedwater valves on affected loop.", tier: "CRITICAL", timeLimitSec: 20 },
        { id: 2, step: "2.0", title: "Depressurize Primary Circuit via Pressurizer Spray", desc: "Equalize primary and secondary pressures to terminate inter-loop radioactive leakage.", tier: "HIGH", timeLimitSec: 60 },
        { id: 3, step: "3.0", title: "Start Radiation Monitoring on Secondary Steam Lines", desc: "Sample condenser off-gas and steam line monitors for N-16 / noble gas activity.", tier: "MANDATORY", timeLimitSec: 90 },
        { id: 4, step: "4.0", title: "Initiate Primary Cooldown via Intact Steam Generator", desc: "Dump steam from intact steam generator at controlled rate (<55°C/hr).", tier: "CONDITIONAL", timeLimitSec: 180 }
      ]
    },
    4: {
      code: "EOP-E4-SBO",
      title: "Station Blackout (SBO) & Natural Circulation Thermosyphon",
      titleHindi: "स्टेशन ब्लैकआउट आपातकालीन शीतलन एवं थर्मोसाइफन संवहन (SBO)",
      priority: "CRITICAL",
      urgencyLevel: "NATURAL CONVECTION COOLING",
      timeToBreachSeconds: 60,
      actions: [
        { id: 1, step: "1.0", title: "Verify Passive Gravity SCRAM", desc: "Verify all control rods fully inserted by gravity upon total loss of AC bus power.", tier: "CRITICAL", timeLimitSec: 5 },
        { id: 2, step: "2.0", title: "Actuate Turbine-Driven Auxiliary Feedwater (TDAFW)", desc: "Start steam-driven feedwater pump requiring zero electrical power for steam generators.", tier: "CRITICAL", timeLimitSec: 25 },
        { id: 3, step: "3.0", title: "Establish Natural Circulation Thermosyphon Flow", desc: "Confirm core ΔT and primary flow stabilize at ~14-16 kg/s via natural thermal buoyancy.", tier: "MANDATORY", timeLimitSec: 60 },
        { id: 4, step: "4.0", title: "Conserve DC Battery Stations & Call Black-Start Diesel", desc: "Shed non-essential DC loads to preserve vital instrumentation for 8+ hours.", tier: "HIGH", timeLimitSec: 120 }
      ]
    },
    5: {
      code: "EOP-E5-ATWS",
      title: "Anticipated Transient Without Scram (ATWS) Emergency Boration",
      titleHindi: "आपातकालीन बोरोनेशन तंत्र एवं दबाव नियंत्रण (ATWS)",
      priority: "CRITICAL",
      urgencyLevel: "EMERGENCY BORATION",
      timeToBreachSeconds: 15,
      actions: [
        { id: 1, step: "1.0", title: "Manual Rod Insertion & Trip Motor-Generator Breakers", desc: "De-energize control rod drive power supplies directly from main breaker panels.", tier: "CRITICAL", timeLimitSec: 5 },
        { id: 2, step: "2.0", title: "Initiate Emergency Boration System", desc: "Start high-pressure boric acid injection pumps to inject 4000 ppm boron solution.", tier: "CRITICAL", timeLimitSec: 15 },
        { id: 3, step: "3.0", title: "Trip Turbine & Verify Secondary Heat Relief", desc: "Open atmospheric relief valves to prevent severe primary overpressure surge.", tier: "MANDATORY", timeLimitSec: 30 }
      ]
    }
  };

  /**
   * Evaluates current plant conditions and returns dynamic, prioritized EOP procedures.
   * @param {number} scenarioId - Classified EOP scenario ID (0-5)
   * @param {Object} telemetry - Current plant state snapshot
   * @param {number|null} criticalTTL - Calculated time to threshold in seconds
   * @returns {Object} Structured EOP guidance with ranked steps and time-to-breach
   */
  function getRankedGuidance(scenarioId, telemetry, criticalTTL) {
    var eop = EOP_DATABASE[scenarioId] || EOP_DATABASE[0];
    var ttl = (criticalTTL !== null && criticalTTL !== undefined && criticalTTL < 900)
      ? criticalTTL
      : eop.timeToBreachSeconds;

    // Clone actions and compute dynamic countdown urgency
    var rankedActions = eop.actions.map(function (act) {
      var remainingSec = Math.max(0, Math.round(ttl * (act.timeLimitSec / eop.timeToBreachSeconds)));
      var status = "PENDING";
      if (scenarioId === 0) status = "ACTIVE";

      return {
        id: act.id,
        step: act.step,
        title: act.title,
        desc: act.desc,
        tier: act.tier,
        timeLimitSec: act.timeLimitSec,
        countdownSec: remainingSec,
        status: status
      };
    });

    return {
      scenarioId: scenarioId,
      code: eop.code,
      title: eop.title,
      titleHindi: eop.titleHindi,
      priority: eop.priority,
      urgencyLevel: eop.urgencyLevel,
      timeToBreachSeconds: ttl,
      actions: rankedActions
    };
  }

  return {
    getRankedGuidance: getRankedGuidance,
    EOP_DATABASE: EOP_DATABASE
  };

})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PrajnaEOPRules;
}
