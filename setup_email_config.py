#!/usr/bin/env python3
"""
Email Configuration Setup for LAEF Trading System
Helps configure email settings for daily reports
"""

import os
import getpass
from pathlib import Path

def setup_email_configuration():
    """Interactive setup for email configuration"""
    
    print("=" * 60)
    print("LAEF EMAIL CONFIGURATION SETUP")
    print("=" * 60)
    print("\nThis will help you configure email reporting for daily trading reports.")
    print("The configuration will be saved to your .env file.\n")
    
    # Read existing .env
    env_file = Path('.env')
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    # Email configuration
    print("EMAIL CONFIGURATION")
    print("-" * 40)
    
    # Enable email
    current_enabled = env_vars.get('EMAIL_ENABLED', 'false')
    enable_email = input(f"Enable email reports? (y/n) [{current_enabled}]: ").strip().lower()
    if enable_email in ['y', 'yes', '1', 'true']:
        env_vars['EMAIL_ENABLED'] = 'true'
    else:
        env_vars['EMAIL_ENABLED'] = 'false'
        print("Email reporting disabled. Skipping email setup.")
        save_env_file(env_vars, env_file)
        return
    
    # SMTP Server
    current_smtp = env_vars.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_server = input(f"SMTP Server [{current_smtp}]: ").strip()
    env_vars['SMTP_SERVER'] = smtp_server or current_smtp
    
    # SMTP Port
    current_port = env_vars.get('SMTP_PORT', '587')
    smtp_port = input(f"SMTP Port [{current_port}]: ").strip()
    env_vars['SMTP_PORT'] = smtp_port or current_port
    
    # From email
    current_from = env_vars.get('EMAIL_FROM', '')
    email_from = input(f"From Email Address [{current_from}]: ").strip()
    env_vars['EMAIL_FROM'] = email_from or current_from
    
    # Email password
    if not env_vars.get('EMAIL_PASSWORD'):
        print("\nFor Gmail, use an 'App Password' instead of your regular password.")
        print("Go to: Google Account > Security > 2-Step Verification > App passwords")
        email_password = getpass.getpass("Email Password (hidden input): ")
        env_vars['EMAIL_PASSWORD'] = email_password
    else:
        update_password = input("Update email password? (y/n): ").strip().lower()
        if update_password in ['y', 'yes', '1']:
            email_password = getpass.getpass("Email Password (hidden input): ")
            env_vars['EMAIL_PASSWORD'] = email_password
    
    # To emails
    current_to = env_vars.get('EMAIL_TO', '')
    print(f"\nEnter recipient emails (comma-separated)")
    email_to = input(f"To Email(s) [{current_to}]: ").strip()
    env_vars['EMAIL_TO'] = email_to or current_to
    
    # Save configuration
    save_env_file(env_vars, env_file)
    
    # Test email
    test_email = input("\nTest email configuration now? (y/n): ").strip().lower()
    if test_email in ['y', 'yes', '1']:
        test_email_setup(env_vars)

def save_env_file(env_vars, env_file):
    """Save environment variables to .env file"""
    
    print(f"\nSaving configuration to {env_file}...")
    
    # Preserve comments and structure
    lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Update existing lines or add new ones
    updated_keys = set()
    
    for i, line in enumerate(lines):
        if '=' in line and not line.strip().startswith('#'):
            key = line.split('=', 1)[0].strip()
            if key in env_vars:
                lines[i] = f"{key}={env_vars[key]}\n"
                updated_keys.add(key)
    
    # Add new keys
    for key, value in env_vars.items():
        if key not in updated_keys:
            lines.append(f"{key}={value}\n")
    
    # Write file
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print("Configuration saved successfully!")

def test_email_setup(env_vars):
    """Test email configuration"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from datetime import datetime
    
    try:
        print("\nTesting email configuration...")
        
        # Create test message
        msg = MIMEMultipart()
        msg['From'] = env_vars['EMAIL_FROM']
        msg['To'] = env_vars['EMAIL_TO']
        msg['Subject'] = f"LAEF Email Test - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        test_body = f"""
LAEF Email Configuration Test

This is a test email from your LAEF Trading System.

Configuration:
- SMTP Server: {env_vars['SMTP_SERVER']}
- SMTP Port: {env_vars['SMTP_PORT']}
- From: {env_vars['EMAIL_FROM']}
- To: {env_vars['EMAIL_TO']}

If you received this email, your configuration is working correctly!

Time: {datetime.now()}
"""
        
        msg.attach(MIMEText(test_body, 'plain'))
        
        # Send test email
        with smtplib.SMTP(env_vars['SMTP_SERVER'], int(env_vars['SMTP_PORT'])) as server:
            server.starttls()
            server.login(env_vars['EMAIL_FROM'], env_vars['EMAIL_PASSWORD'])
            server.send_message(msg)
        
        print("✓ Test email sent successfully!")
        print(f"Check your inbox at: {env_vars['EMAIL_TO']}")
        
    except Exception as e:
        print(f"✗ Email test failed: {e}")
        print("\nCommon issues:")
        print("1. Gmail requires 'App Passwords' - not your regular password")
        print("2. Check SMTP server and port settings")
        print("3. Verify email addresses are correct")
        print("4. Some email providers require additional security settings")

def main():
    """Main setup function"""
    print("Starting LAEF Email Configuration Setup...")
    
    try:
        setup_email_configuration()
        
        print("\n" + "=" * 60)
        print("EMAIL SETUP COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the automated daily trader to test full system")
        print("2. Set up Windows Task Scheduler for daily automation")
        print("3. Monitor the first few daily reports to ensure everything works")
        print("\nDaily reports will include:")
        print("- Portfolio performance")
        print("- Trading activity")
        print("- Learning progress")
        print("- Model evolution")
        print("- Strategy insights")
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\nSetup failed: {e}")

if __name__ == "__main__":
    main()