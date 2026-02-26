# Email Configuration

This folder contains configuration files for the CT600 Vision Inspection system.

## email_config.json

This file contains email settings for judgement mismatch alerts.

### Configuration Fields

- **RTO0006**: Application ID (e.g., "CT600")
- **RTO0010**: Email category for filtering (e.g., "judgement_alert")
- **RTO0013_01**: Sender email address
- **RTO0013_02**: Receiver email address(es) - comma-separated for multiple recipients
- **RTO0013_03**: CC email address(es) - comma-separated for multiple recipients (can be empty string)

### Example Configuration

```json
{
    "RTO0006": "CT600",
    "RTO0010": "judgement_alert",
    "RTO0013_01": "ct600@murata.com",
    "RTO0013_02": "admin@murata.com,manager@murata.com",
    "RTO0013_03": "supervisor@murata.com"
}
```

### Notes

- If the config file is missing or invalid, the system will use default values and log a warning
- The configuration is loaded at application startup
- Changes to the config file require restarting the application to take effect
