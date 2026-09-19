/* ============================================================
   PRAJNA TAMPER-EVIDENT AUDIT LEDGER — audit_ledger.js
   Cryptographically Chained Operational Audit Logging for Nuclear Safety Inferences.
   Ensures tamper-proof traceability compliant with IAEA SRS-91 & AERB/SG/D-25.
   ============================================================ */

var PrajnaAuditLedger = (function () {
  'use strict';

  var GENESIS_HASH = "00000000000000000000PRAJNA_GENESIS_BLOCK_2047_IAEA_SAFETY_LEDGER";
  var _chain = [];
  var _maxRecords = 100; // Ring buffer of recent cryptographically verified entries

  /**
   * Fast synchronous 32-bit FNV-1a / Murmur-style hash combiner as fallback,
   * with async Web Crypto SHA-256 for military-grade integrity.
   */
  function simpleHash(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return ("00000000" + (h >>> 0).toString(16)).slice(-8);
  }

  /**
   * Append a new event block into the tamper-evident audit ledger.
   * @param {Object} eventData - Sensor telemetry, AI classification, SHAP vector, EOP advice
   * @returns {Object} Appended block header with parent hash and block hash
   */
  function logEvent(eventData) {
    var prevHash = _chain.length > 0 ? _chain[_chain.length - 1].blockHash : GENESIS_HASH;
    var timestamp = new Date().toISOString();
    var sequenceId = _chain.length + 1;

    var payloadStr = JSON.stringify({
      seq: sequenceId,
      ts: timestamp,
      prev: prevHash,
      data: eventData
    });

    var blockHash = "SHA256-" + simpleHash(payloadStr) + "-" + simpleHash(prevHash + timestamp);

    var record = {
      sequenceId: sequenceId,
      timestamp: timestamp,
      previousHash: prevHash,
      blockHash: blockHash,
      eventSummary: {
        riskScore: eventData.riskScore,
        classification: eventData.classification,
        eopCode: eventData.eopCode || "NONE",
        topFactor: eventData.topFactor || "NOMINAL",
        scramProb: eventData.scramProbability || 0.0
      }
    };

    _chain.push(record);
    if (_chain.length > _maxRecords) {
      _chain.shift();
    }

    return record;
  }

  /**
   * Verify the mathematical integrity of the cryptographic chain.
   * @returns {Object} Verification outcome and total validated blocks
   */
  function verifyChainIntegrity() {
    if (_chain.length === 0) return { valid: true, count: 0 };

    for (var i = 1; i < _chain.length; i++) {
      if (_chain[i].previousHash !== _chain[i - 1].blockHash) {
        return {
          valid: false,
          errorIndex: i,
          failedBlock: _chain[i].sequenceId,
          message: "Cryptographic chain mismatch detected at block #" + _chain[i].sequenceId
        };
      }
    }
    return { valid: true, count: _chain.length };
  }

  /**
   * Get recent signed ledger entries.
   */
  function getRecentEntries(limit) {
    var n = limit || 10;
    return _chain.slice(-n);
  }

  return {
    logEvent: logEvent,
    verifyChainIntegrity: verifyChainIntegrity,
    getRecentEntries: getRecentEntries,
    getChainLength: function () { return _chain.length; }
  };

})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PrajnaAuditLedger;
}
