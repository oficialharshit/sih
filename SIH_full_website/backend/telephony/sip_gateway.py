"""
SIP Gateway & Protocol Analyzer for voxguard.
Implements RFC 3261 SIP packet parsing, caller identity extraction,
and realistic enterprise PBX/telecom scenarios for fraud interception.
"""

import re
import uuid
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("sentinel_sip_gateway")


class SIPHeaderParser:
    """Parses RFC 3261 SIP messages and extracts caller & routing metadata."""

    @staticmethod
    def parse_sip_packet(raw_sip: str) -> Dict[str, Any]:
        """
        Parses a raw text SIP packet (e.g. INVITE request) and extracts
        From, To, Call-ID, CSeq, X-Headers, and SDP audio parameters.
        """
        headers: Dict[str, str] = {}
        sdp_lines = []
        is_sdp = False

        for line in raw_sip.strip().splitlines():
            line = line.strip()
            if not line:
                is_sdp = True
                continue
            if is_sdp:
                sdp_lines.append(line)
            else:
                if ":" in line:
                    key, val = line.split(":", 1)
                    headers[key.strip().upper()] = val.strip()

        from_hdr = headers.get("FROM", "")
        caller_name = "Unknown Caller"
        caller_uri = ""

        name_match = re.search(r'"([^"]+)"', from_hdr)
        if name_match:
            caller_name = name_match.group(1)

        uri_match = re.search(r'<([^>]+)>', from_hdr)
        if uri_match:
            caller_uri = uri_match.group(1)
        elif "@" in from_hdr:
            caller_uri = from_hdr

        extension = headers.get("X-PBX-EXTENSION", "")
        if not extension and caller_uri:
            user_part = caller_uri.replace("sip:", "").split("@")[0]
            if user_part.isdigit():
                extension = user_part

        audio_codec = "PCMU/8000"
        for sdp in sdp_lines:
            if sdp.startswith("a=rtpmap:"):
                audio_codec = sdp.replace("a=rtpmap:", "").split(" ", 1)[-1]
                break

        return {
            "method": raw_sip.split()[0] if raw_sip else "INVITE",
            "caller_name": caller_name,
            "caller_uri": caller_uri,
            "extension": extension or "4042",
            "call_id": headers.get("CALL-ID", str(uuid.uuid4())),
            "trunk_id": headers.get("X-PBX-TRUNK", "CISCO-CUCM-MUMBAI-01"),
            "codec": audio_codec,
            "headers": headers,
            "raw_packet": raw_sip
        }


SIP_DEMO_SCENARIOS = [
    {
        "id": "ceo_impersonation_scam",
        "title": "🚨 CEO Impersonation (High Risk Attack)",
        "caller_name": "Rajesh Sharma",
        "caller_title": "Group Chief Executive Officer",
        "extension": "4001",
        "claimed_identity": "ceo_rajesh_sharma",
        "environment": "sip",
        "pbx_trunk": "AVAYA-AURA-HQ-01",
        "sample_audio": "sample.wav",
        "raw_sip": (
            "INVITE sip:treasury.desk@hdfc-vault.internal:5060 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.20.1.55:5060;branch=z9hG4bK-771829\r\n"
            "From: \"Rajesh Sharma (CEO)\" <sip:rajesh.sharma@hdfc-vault.internal>;tag=991823\r\n"
            "To: \"Treasury Desk\" <sip:treasury.desk@hdfc-vault.internal>\r\n"
            "Call-ID: sip-8942-ceo-urgent-auth-cucm@10.20.1.55\r\n"
            "CSeq: 101 INVITE\r\n"
            "X-PBX-Extension: 4001\r\n"
            "X-PBX-Trunk: AVAYA-AURA-HQ-01\r\n"
            "X-Identity-Asserted: True (Internal IP-PBX)\r\n"
            "Content-Type: application/sdp\r\n\r\n"
            "v=0\r\n"
            "o=RajeshSharma 2890844526 2890844526 IN IP4 10.20.1.55\r\n"
            "s=Urgent RTGS Authorization Call\r\n"
            "c=IN IP4 10.20.1.55\r\n"
            "t=0 0\r\n"
            "m=audio 49170 RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
        ),
        "description": "Incoming VoIP call spoofing executive extension 4001 demanding an urgent INR 2.5 Crore RTGS clearance."
    },
    {
        "id": "cfo_vendor_payment",
        "title": "🚨 CFO Vendor Spoofing (Urgent Invoice)",
        "caller_name": "Priya Nair",
        "caller_title": "Chief Financial Officer",
        "extension": "4005",
        "claimed_identity": "cfo_priya_nair",
        "environment": "sip",
        "pbx_trunk": "CISCO-CUCM-FIN-02",
        "sample_audio": "sample.wav",
        "raw_sip": (
            "INVITE sip:accounts.payable@hdfc-vault.internal:5060 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.20.2.14:5060;branch=z9hG4bK-449102\r\n"
            "From: \"Priya Nair (CFO)\" <sip:priya.nair@hdfc-vault.internal>;tag=110293\r\n"
            "To: \"Accounts Payable\" <sip:accounts.payable@hdfc-vault.internal>\r\n"
            "Call-ID: sip-3301-cfo-vendor-override@10.20.2.14\r\n"
            "CSeq: 204 INVITE\r\n"
            "X-PBX-Extension: 4005\r\n"
            "X-PBX-Trunk: CISCO-CUCM-FIN-02\r\n"
            "Content-Type: application/sdp\r\n\r\n"
            "v=0\r\n"
            "o=PriyaNair 3019284 3019284 IN IP4 10.20.2.14\r\n"
            "s=Vendor Payment Override\r\n"
            "c=IN IP4 10.20.2.14\r\n"
            "t=0 0\r\n"
            "m=audio 5004 RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
        ),
        "description": "VoIP SIP call spoofing CFO Priya Nair demanding emergency payment override for vendor software licensing."
    },
    {
        "id": "contact_center_otp_fraud",
        "title": "🏢 Contact Center Customer Impersonation",
        "caller_name": "Unknown Customer (+91-98201-XXXXX)",
        "caller_title": "Retail Banking Caller",
        "extension": "Queue-IVR-100",
        "claimed_identity": "auto",
        "environment": "contact_center",
        "pbx_trunk": "GENESYS-CLOUD-DELHI",
        "sample_audio": "sample.wav",
        "raw_sip": (
            "INVITE sip:support.agent@bank-contact-center.in:5060 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 172.16.8.20:5060;branch=z9hG4bK-cc9921\r\n"
            "From: \"Caller: +919820123456\" <sip:customer@gateway.telco.in>;tag=7719\r\n"
            "To: \"Tier 1 Agent\" <sip:agent44@bank-contact-center.in>\r\n"
            "Call-ID: cc-ivr-genesys-tap-099281@172.16.8.20\r\n"
            "X-Contact-Center-Queue: HIGH_VALUE_CLIENTS\r\n"
            "X-Audio-Fork: DUAL_CHANNEL_TAP\r\n"
            "Content-Type: application/sdp\r\n\r\n"
            "v=0\r\n"
            "o=GenesysFork 99120 99120 IN IP4 172.16.8.20\r\n"
            "s=Contact Center Media Stream Fork\r\n"
            "c=IN IP4 172.16.8.20\r\n"
            "m=audio 16000 RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
        ),
        "description": "Incoming inbound call fork from contact center queue attempting social engineering to bypass 2FA."
    },
    {
        "id": "zoom_executive_conference",
        "title": "👥 Collaboration Meeting (Zoom Boardroom)",
        "caller_name": "Rajesh Sharma",
        "caller_title": "Boardroom Participant",
        "extension": "Zoom-Room-04",
        "claimed_identity": "ceo_rajesh_sharma",
        "environment": "zoom",
        "pbx_trunk": "ZOOM-ROOM-SIP-CONNECTOR",
        "sample_audio": "sample.wav",
        "raw_sip": (
            "INVITE sip:boardroom@zoomcrc.com:5060 SIP/2.0\r\n"
            "Via: SIP/2.0/TLS 198.51.100.22:5061;branch=z9hG4bK-zm4912\r\n"
            "From: \"Rajesh Sharma (Zoom Audio)\" <sip:participant@zoom.us>;tag=zm9821\r\n"
            "To: \"Quarterly Audit Conference\" <sip:audit@zoomcrc.com>\r\n"
            "Call-ID: zoom-sip-connector-conf-9921401@198.51.100.22\r\n"
            "X-Collaboration-Platform: Zoom Meeting SDK / CRC\r\n"
            "Content-Type: application/sdp\r\n\r\n"
            "v=0\r\n"
            "o=ZoomAudio 10029 10029 IN IP4 198.51.100.22\r\n"
            "s=Quarterly Audit Call\r\n"
            "m=audio 24000 RTP/AVP 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
        ),
        "description": "Audio feed intercepted from executive video conference room testing for real-time deepfake voice cloning."
    }
]
