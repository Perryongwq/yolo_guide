import smtplib, os, sys, json
import pandas as pd
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
format = "%Y/%m/%d %H:%M:%S"


class sendmail:
    def __init__(self, mto="", mfrom="", mcc="", applicationid=""):
        self.mhosts = "172.24.128.80"
        self.mto = mto
        self.mfrom = mfrom
        self.mcc = mcc
        self.applicationid = f"{applicationid}.mailsender"

    def sendmailhtmlformat(self, msubject, messages):
        msg = MIMEMultipart()
        msg["Subject"] = msubject
        msg["From"] = self.mfrom
        msg["To"] = self.mto
        msg["Cc"] = self.mcc
        msg.attach(MIMEText(messages, "html"))

        server = smtplib.SMTP(self.mhosts)
        try:
            server.send_message(msg)
            print(f"Email sent successfully: {msubject}")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            cdatetime = datetime.now().strftime(format)
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            errmsg = f"{cdatetime} : {exc_type} : {fname} : {exc_tb.tb_lineno} : {e}"
            logger.error(errmsg)
            print(f"Failed to send email: {errmsg}")
        finally:
            server.quit()


def program_error_email_content(error_type="System Error", source_name=None, details=None):
    """
    Generate email content for NCR alerts.

    Args:
        error_type (str): Type of system error
        source_name (str): Name of the system/module that encountered the error
        details (dict or str): Additional details about the error

    Returns:
        tuple: (subject, html_body)
    """
    if source_name is None:
        source_name = "BAC Application"

    subject = f"System Alert: {error_type}"

    description = f"""
        <p>This is an automated system alert to notify you that a Misjudgement case has been detected in</p>
        <p><b>{source_name} - Check the Lot information</b></p>
    """

    if details:
        if isinstance(details, dict):
            description += "<ul>"
            for key, value in details.items():
                description += f"<li><b>{key}:</b> {value}</li>"
            description += "</ul>"
        else:
            description += f"<p>{details}</p>"

    description += """
        <p><b>Action Required:</b> Please investigate and issue NCR and take appropriate action</p>
        <p>Thank you.</p>
    """

    return subject, description


def draftcontentandsend(emailq, subject, body, application_id):
    """Wrap body in HTML and send."""
    try:
        html_content = f"""
        <html>
        <body>
            {body}
        </body>
        </html>
        """
        emailq.sendmailhtmlformat(str(subject), str(html_content))
    except Exception as er:
        logger.error(f"Error in draftcontentandsend [{application_id}]: {er}")
        print(f"Error in draftcontentandsend: {er}")


def sendemail(subject, body, category, emaildf, application_id):
    """Send email to all recipients matching the given category in the DataFrame."""
    try:
        logger.info(f"[sendemail] Looking for category: '{category}'")
        logger.info(f"[sendemail] Email DataFrame columns: {list(emaildf.columns)}")
        logger.info(f"[sendemail] Email DataFrame RTO0010 values: {emaildf['RTO0010'].tolist() if 'RTO0010' in emaildf.columns else 'RTO0010 column not found'}")
        
        emailq = sendmail()
        filtered_df = emaildf[emaildf["RTO0010"] == category]
        
        logger.info(f"[sendemail] Filtered DataFrame rows: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            error_msg = f"No email configuration found for category '{category}'. Available categories: {emaildf['RTO0010'].unique().tolist() if 'RTO0010' in emaildf.columns else 'N/A'}"
            logger.error(f"[sendemail] {error_msg}")
            print(f"[sendemail] {error_msg}")
            raise ValueError(error_msg)

        for _, row in filtered_df.iterrows():
            logger.info(f"[sendemail] Sending email to: {row['RTO0013_02']}, from: {row['RTO0013_01']}")
            emailq.__init__(
                mto=row["RTO0013_02"],
                mfrom=row["RTO0013_01"],
                mcc=row["RTO0013_03"],
                applicationid=row["RTO0006"],
            )
            draftcontentandsend(emailq, subject, body, application_id)
            logger.info(f"[sendemail] Email sent successfully")
    except Exception as er:
        logger.error(f"Error in sendemail [{application_id}]: {er}", exc_info=True)
        print(f"Error in sendemail: {er}")
        raise


def sendErrorAlertEmail(emaildf, email_category, error_type="System Error", app_name=None, details=None):
    """
    Simple email alert function.

    Args:
        emaildf (pd.DataFrame): DataFrame with email config columns:
            - RTO0006:     Application ID
            - RTO0010:     Email category (used to filter rows)
            - RTO0013_01:  Sender email
            - RTO0013_02:  Receiver email(s), comma-separated
            - RTO0013_03:  CC email(s), comma-separated
        email_category (str): Category to filter the DataFrame (e.g. 'sendemail')
        error_type (str): Error type / subject text
        app_name (str): Application name for the alert
        details (dict or str, optional): Extra error details
    """
    try:
        subject, body = program_error_email_content(error_type, app_name, details)
        sendemail(subject, body, email_category, emaildf, app_name)
    except Exception as e:
        logger.error(f"Error in sendErrorAlertEmail: {e}")
        print(f"Error in sendErrorAlertEmail: {e}")


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":

    ApplicationID = "BAC "

    # Email configuration data
    data = [
        {
            "RTO0006": ApplicationID,                                       # App ID
            "RTO0010": "sendemail",                                         # Email category
            "RTO0013_01": "sp5143@murata.com",                              # Sender
            "RTO0013_02": "perry.ong@murata.com",   # Receivers
            "RTO0013_03": "perry.ong@murata.com",  # CC
        }
    ]

    emaildf = pd.DataFrame(data)

    # Send a simple error alert email
    sendErrorAlertEmail(
        emaildf,
        "sendemail",
        f"NCR Alert {ApplicationID}",
        ApplicationID,
    )
