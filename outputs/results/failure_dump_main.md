# main: 200 test items joined

## MISSED HARD ESCALATIONS
- [A|aut] '@SpotifyCares hi, sometimes when I go back on Spotify a song I would never play is on the currently playing.'
    label account_access_security | ESC:account_or_pii   pred playback_or_app_issue | auto (model auto, forced None) sim 0.32
    reply "Hey! Can you let us know the device and Spotify version you're using? We'll see what we can suggest. Also, try logging out, restarting your device, an"
- [A|ass] "@SpotifyCares hello I'm trying to sign up for the spotify premium for 3 months THING but every every time I put my bank details and click next it says the site is timed o"
    label payment_billing_subscription | ESC:payment_refund   pred payment_billing_subscription | auto (model auto, forced None) sim 0.25
    reply 'Hey there! Can you try signing up using an incognito window or a different browser instead? Let us know how it goes'
- [B|ass] '@SpotifyCares I’ve been constantly trying to delete songs in my library but they always seem to come back. I’ve tried already the connect/disconnect method that I was fir'
    label library_or_playlist_problem +support_followup_or_complaint | ESC:explicit_human_request   pred library_or_playlist_problem | auto (model auto, forced None) sim 0.17
    reply "Hi there! We'd love to help. Can you let us know what's happening exactly? We'll see what we can suggest"
- [B|ass] '@115888 I just made an account and for some reason my username was automatically made my password??'
    label account_access_security | ESC:account_or_pii   pred account_settings_change | auto (model auto, forced None) sim 0.35
    reply 'Hey there! Your username is automatically generated when you sign up. Your display name (what shows in your profile) can be changed later. Let us know'
- [B|ass] '@117153\nWhy does my music keep getting cut off on my office pc when my wife listens to Spotify in the house🤔\n\nWe are paying Premium for Family Acct. \n\nWe have separate si'
    label family_or_student_plan +playback_or_app_issue | ESC:account_or_pii   pred playback_or_app_issue | auto (model auto, forced None) sim 0.2
    reply "Hey there! Can you let us know what device, operating system, and Spotify version you're using on your office PC? We'll see what we can suggest."
- [B|ass] '@148611 hello why am i banned from reading the spotify community page :('
    label account_access_security | ESC:account_or_pii   pred content_request_or_availability | auto (model auto, forced None) sim 0.27
    reply "Hey! We're here to help. Can you let us know what happens when you try to access the Spotify Community page? We'll check it out and get back to you."
- [B|ass] "@SpotifyCares Hi there! I lost my NUS discount since the change to UNIDAYS - they don't recognise my institution; is there anything you can do?"
    label family_or_student_plan | ESC:account_or_pii   pred family_or_student_plan | auto (model auto, forced None) sim 0.39
    reply 'Hi there! NUS is currently unavailable as a discount option. We’ll pass your feedback to the team. For now, you can try using UNiDAYS at https://t.co/'

## MISSED SOFT/ROUTING ESCALATIONS
- [B|aut] '@SpotifyCares wat is you doin https://t.co/xYZ1EqGige'
    label other_unclear | ESC:no_actionable_content   pred content_request_or_availability | auto (model auto, forced None) sim 0.36
- [A|ass] 'Wtf is this lmao @115888 @116602 @118134 https://t.co/Z5ba6HJ7ED'
    label other_unclear | ESC:no_actionable_content   pred content_request_or_availability | auto (model auto, forced None) sim 0.65
- [A|ass] '@115888 came out with a great promotion — Hulu &amp; Spotify Premium for students at a low cost of $4.99 per month. WELL, that was false advertisement ... #wtf \n🚮'
    label family_or_student_plan | ESC:anger_churn   pred content_request_or_availability | auto (model auto, forced None) sim 0.24
- [A|ass] 'Con Premium, escucha música sin conexión, estés donde estés. 3 meses por 0.99 USD. https://t.co/7OWLY9NaKI'
    label other_unclear | ESC:non_english   pred content_request_or_availability | auto (model auto, forced None) sim 0.22

## FALSE ESCALATIONS (label auto, pred escalate)
- [A|ass] 'thanks boo @SpotifyCares https://t.co/wlc0dCsTKd'
    label other_unclear | auto   pred feature_request_or_feedback | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.49
- [A|aut] "So @115888 you kick me off my families Family Plan then your website isn't allowing me to accept the invitation while trying to get back on the plan, why?"
    label family_or_student_plan | auto   pred family_or_student_plan | escalate:account_or_pii (model escalate, forced None) sim 0.3
- [A|ass] '@115888 should publish a “Shuffle All” feature that shuffles all of Your Daily Mixes\n\n@125633 @SpotifyCares @117168'
    label feature_request_or_feedback | auto   pred feature_request_or_feedback | escalate:draft_invalid (model auto, forced draft_invalid) sim 0.28
- [B|ass] '@SpotifyCares Should I be worried about having a lot of my saved music under my "Liked from Radio" playlist? Is that ever in danger of being deleted/changed/something?'
    label library_or_playlist_problem | auto   pred library_or_playlist_problem | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.22
- [A|ass] 'Appreciate it. https://t.co/7BxCc61d9G'
    label other_unclear | auto   pred feature_request_or_feedback | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.53
- [B|ass] '@117153 Hello. I am having issues with my Spotify family premium.'
    label family_or_student_plan | auto   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.53
- [A|ass] '@SpotifyCares having some issues'
    label support_followup_or_complaint | auto   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.56
- [A|ass] "@SpotifyCares - any chance you can help? I didn't realise how much I rely on Spotify until I didn't have it any more :( https://t.co/hT2JJxSEsH"
    label support_followup_or_complaint | auto   pred playback_or_app_issue | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.26
- [B|ass] '@SpotifyCares I need answers pls. 🤔 https://t.co/gwJDWDImoW'
    label support_followup_or_complaint | auto   pred account_access_security | escalate:explicit_human_request (model escalate, forced None) sim 0.69
- [A|ass] '@SpotifyCares big ups to your online chat support agent Ronald D for helping me get my billing info off my ex’s account! Quick and painless! Kudos to Ronald! 🍻'
    label support_followup_or_complaint | auto   pred support_followup_or_complaint | escalate:payment_refund (model auto, forced payment_refund) sim 0.23
- [A|ass] '@SpotifyCares Hey pals, I made a new account and it has my username as a bunch of random letters and numbers, when I put in an actual name for the username when making th'
    label account_settings_change | auto   pred account_settings_change | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.44
- [A|ass] 'Why the hell does @115888 care in which country I am? This is dumb every time I travel I need to change or upgrade?'
    label account_settings_change | auto   pred account_settings_change | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.23
- [A|ass] '@SpotifyCares Your account overview page keeps 500’ing on me: https://t.co/ADn6Fc1vY9'
    label playback_or_app_issue | auto   pred account_access_security | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.3
- [A|ass] '@125633 how do you change your Artist name on Spotify?'
    label account_settings_change | auto   pred account_settings_change | escalate:draft_invalid (model auto, forced draft_invalid) sim 0.46
- [B|ass] '@SpotifyCares Hi! I just want to inquire if I can avail the 9 pesos for 3 months promo if the mode of payment is thru paymaya. And can I still avail if my account is not '
    label payment_billing_subscription | auto   pred payment_billing_subscription | escalate:payment_refund (model auto, forced dm_handoff) sim 0.2

## INTENT ERRORS (strict)
- [A|ass] 'thanks boo @SpotifyCares https://t.co/wlc0dCsTKd'
    label other_unclear | auto   pred feature_request_or_feedback | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.49
- [B|ass] '@115888 I’ve been trying to update my payment method on you website for the last 45 minutes, but it won’t work. I bet @129941 would take my payment/new card info.'
    label payment_billing_subscription | ESC:payment_refund   pred account_access_security | escalate:account_or_pii (model escalate, forced payment_refund) sim 0.35
- [B|aut] '@SpotifyCares wat is you doin https://t.co/xYZ1EqGige'
    label other_unclear | ESC:no_actionable_content   pred content_request_or_availability | auto (model auto, forced None) sim 0.36
- [B|aut] "@SpotifyCares me and my friend made a song and my name is under the right profile but his isn't can you fix that?"
    label content_request_or_availability | auto   pred account_settings_change | auto (model auto, forced None) sim 0.27
- [B|aut] '@115888 you mean to tell me I pay 10 dollars a month for you to have only 3 RBD songs? I’m heartbroken https://t.co/siBTlHyYfA'
    label content_request_or_availability | auto   pred playback_or_app_issue | auto (model auto, forced None) sim 0.28
- [A|aut] '@SpotifyCares hi, sometimes when I go back on Spotify a song I would never play is on the currently playing.'
    label account_access_security | ESC:account_or_pii   pred playback_or_app_issue | auto (model auto, forced None) sim 0.32
- [A|aut] '@115888 I think you need to have your security team look into this.\nCc: @123321 @466175 https://t.co/kGwvGpZ0PK'
    label other_unclear | ESC:legal_safety_threat   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.3
- [A|aut] '@SpotifyCares I can’t get Hulu on my premium for students + Hulu account. When I try to log into Hulu I get this message “we’re having trouble authorizing your account wi'
    label family_or_student_plan | ESC:account_or_pii   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.73
- [A|aut] 'Har stört mig länge nu på att ni har skrivit "Ditt Bibliotek", "Din daily mix", "Din musik2. Är det inte tänkt att det ska vara Mitt Bibliotek när det är just min musik, '
    label feature_request_or_feedback | ESC:non_english   pred library_or_playlist_problem | escalate:non_english (model auto, forced non_english) sim 0.39
- [A|aut] '@642 @148611 san na po ang PIN number??? Nakailang ulit ko na to https://t.co/NTSEhz4Wvf'
    label account_access_security | ESC:non_english   pred content_request_or_availability | escalate:non_english (model auto, forced non_english) sim 0.32
- [A|aut] "@SpotifyCares  i can't join the Spotify Premium, that i had use it for a while but i can't use it since last week! please help"
    label payment_billing_subscription | ESC:payment_refund   pred account_access_security | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.24
- [B|aut] '@115888 I recently updated my account details to claim student discount, however I’m now being charged for Premium and the student price! Please can you help me! 😞'
    label payment_billing_subscription +family_or_student_plan | ESC:payment_refund   pred family_or_student_plan | escalate:payment_refund (model escalate, forced payment_refund) sim 0.3
- [A|aut] '@127637 des soucis avec la connection @118117 ? Impossible de me reconnecter à ma Play:1 Cc @SpotifyCares @137949'
    label playback_or_app_issue | ESC:non_english   pred account_access_security | escalate:non_english (model auto, forced non_english) sim 0.31
- [A|ass] "Does anyone know why @115888 isn't working?"
    label playback_or_app_issue | auto   pred content_request_or_availability | auto (model auto, forced None) sim 0.5
- [A|ass] "I've picked up on @115888 shuffle algorithm and I just want it to be genuinely random. Can you turn off your algorithm for me? Thanks"
    label library_or_playlist_problem | auto   pred feature_request_or_feedback | auto (model auto, forced None) sim 0.39
- [B|ass] '@SpotifyCares hi after of 3 months, how much money would have I pay to listen music? Any feature of Spotify app is free? Why you make services have a cost? Tell me motive'
    label payment_billing_subscription | auto   pred feature_request_or_feedback | auto (model auto, forced None) sim 0.33
- [A|ass] 'Hey @115888 would ❤️ you forever if you could put out a "Baby, It\'s Cold Outside"-free holiday playlist \n\nx, team SG'
    label feature_request_or_feedback | auto   pred content_request_or_availability | auto (model auto, forced None) sim 0.3
- [A|ass] 'Fwd: @spotifycares https://t.co/U3XE32TsG4'
    label other_unclear | ESC:no_actionable_content   pred feature_request_or_feedback | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.83
- [B|ass] '@SpotifyCares why was I not notified about needing to re-verify my student account eligibility? I tried to get the student account again and the website now says I’m a st'
    label family_or_student_plan | ESC:account_or_pii   pred payment_billing_subscription | escalate:payment_refund (model escalate, forced None) sim 0.25
- [A|ass] 'Appreciate it. https://t.co/7BxCc61d9G'
    label other_unclear | auto   pred feature_request_or_feedback | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.53
- [B|ass] "@117153 I'm premium but the commercials haven't stop. Is it something wrong with Spotify?"
    label payment_billing_subscription | ESC:payment_refund   pred playback_or_app_issue | escalate:payment_refund (model auto, forced payment_refund) sim 0.32
- [A|ass] 'Are you kidding me @115888 ? In a 50 minute period I got nine minutes of music and the rest was ads come on please stop this I know you guys gotta make money but come on '
    label feature_request_or_feedback | auto   pred playback_or_app_issue | auto (model auto, forced None) sim 0.31
- [A|ass] '@115888 is it a feature to use my password as the very prominently displayed and in clear text username in the app now?   #WTF #netsec miss'
    label account_access_security | ESC:account_or_pii   pred account_settings_change | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.2
- [A|ass] 'I’d like to see my @115888 play count for both Twinkle Twinkle Little Star (the hit single) and the album “2017 White Noise for Baby Sleep” #breakingrecords #repeatONE'
    label feature_request_or_feedback | auto   pred content_request_or_availability | auto (model auto, forced None) sim 0.17
- [B|ass] '@117153 Hello. I am having issues with my Spotify family premium.'
    label family_or_student_plan | auto   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.53
- [B|ass] '@115888 I got a new debit card and am trying to change my payment information for my account. It’s not allowing me to change anything though. Help'
    label payment_billing_subscription | ESC:payment_refund   pred account_settings_change | escalate:payment_refund (model auto, forced payment_refund) sim 0.34
- [A|ass] '@SpotifyCares I have the Student discount. I wanted to update my payment method but it shows the full price of 9.99. What should I do?'
    label family_or_student_plan +payment_billing_subscription | ESC:payment_refund   pred payment_billing_subscription | escalate:payment_refund (model escalate, forced payment_refund) sim 0.35
- [A|ass] '@SpotifyCares having some issues'
    label support_followup_or_complaint | auto   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.56
- [A|ass] "@SpotifyCares - any chance you can help? I didn't realise how much I rely on Spotify until I didn't have it any more :( https://t.co/hT2JJxSEsH"
    label support_followup_or_complaint | auto   pred playback_or_app_issue | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.26
- [A|ass] "@SpotifyCares hello quick question - where do i sign up to get end of year stats? BECAUSE I MISSED OUT LAST TIME :'("
    label feature_request_or_feedback | auto   pred content_request_or_availability | auto (model auto, forced None) sim 0.41
- [A|ass] 'Wtf is this lmao @115888 @116602 @118134 https://t.co/Z5ba6HJ7ED'
    label other_unclear | ESC:no_actionable_content   pred content_request_or_availability | auto (model auto, forced None) sim 0.65
- [A|ass] '@SpotifyCares uh https://t.co/snYdK8YxpJ'
    label other_unclear | ESC:no_actionable_content   pred playback_or_app_issue | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.75
- [B|ass] '@SpotifyCares \nI had fraud on my bank account in August 2016. Since then a subscription of 9.99 for spotify has been coming out of my bank account. I do not have a spotif'
    label payment_billing_subscription | ESC:payment_refund   pred account_access_security | escalate:account_or_pii (model escalate, forced legal_safety_threat) sim 0.26
- [A|ass] '@SpotifyCares Our family account seems to no longer exist. I have only spotify free and paypal payment not taken. Can you advise?'
    label family_or_student_plan +payment_billing_subscription | ESC:payment_refund   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.25
- [A|ass] '@SpotifyCares accidentally revoked a 3rd party app and i can’t re-gain access to it. how do i gain access to it again??'
    label account_settings_change | auto   pred account_access_security | auto (model auto, forced None) sim 0.26
- [B|ass] "@117168 @SpotifyCares @115888 is there a valid reason why I can't edit the order of my playlists on my Android device? That should really be something catered to all devi"
    label feature_request_or_feedback | auto   pred playback_or_app_issue | auto (model auto, forced None) sim 0.28
- [A|ass] '@115888 u ok?'
    label other_unclear | ESC:no_actionable_content   pred support_followup_or_complaint | escalate:no_actionable_content (model auto, forced no_actionable_content) sim 0.66
- [A|ass] 'Hi @SpotifyCares  I just sent you DM regarding payment'
    label support_followup_or_complaint +payment_billing_subscription | ESC:explicit_human_request   pred payment_billing_subscription | escalate:non_english (model escalate, forced non_english) sim 0.69
- [B|ass] '@115888 I just made an account and for some reason my username was automatically made my password??'
    label account_access_security | ESC:account_or_pii   pred account_settings_change | auto (model auto, forced None) sim 0.35
- [A|ass] '@115888 came out with a great promotion — Hulu &amp; Spotify Premium for students at a low cost of $4.99 per month. WELL, that was false advertisement ... #wtf \n🚮'
    label family_or_student_plan | ESC:anger_churn   pred content_request_or_availability | auto (model auto, forced None) sim 0.24
- [B|ass] "Still nothing, even when I'm on WiFi.\nAny solution to this, @115888?\nSeems like a fatal flaw in your login system. https://t.co/p2YuhUtVoK"
    label account_access_security | ESC:account_or_pii   pred playback_or_app_issue | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.23
- [A|ass] '@SpotifyCares just signed up for a family account but want to switch to a student account - how can i go about doing that?'
    label payment_billing_subscription +family_or_student_plan | auto   pred family_or_student_plan | auto (model auto, forced None) sim 0.37
- [B|ass] '@SpotifyCares How to add an account to a new phone as part of family plan?'
    label family_or_student_plan | auto   pred account_settings_change | auto (model auto, forced None) sim 0.39
- [B|ass] '@SpotifyCares please reply to my private message? Been charged when I shouldn’t have been'
    label payment_billing_subscription +support_followup_or_complaint | ESC:payment_refund   pred account_access_security | escalate:payment_refund (model escalate, forced payment_refund) sim 0.38
- [A|ass] "@SpotifyCares i want to switch to spotify premium with Indosat, I already entering my number but I don't get the pin code. Help please :("
    label payment_billing_subscription | ESC:payment_refund   pred account_access_security | escalate:draft_invalid (model auto, forced draft_invalid) sim 0.29
- [B|ass] '@SpotifyCares I need answers pls. 🤔 https://t.co/gwJDWDImoW'
    label support_followup_or_complaint | auto   pred account_access_security | escalate:explicit_human_request (model escalate, forced None) sim 0.69
- [A|ass] 'Con Premium, escucha música sin conexión, estés donde estés. 3 meses por 0.99 USD. https://t.co/7OWLY9NaKI'
    label other_unclear | ESC:non_english   pred content_request_or_availability | auto (model auto, forced None) sim 0.22
- [B|ass] '@117153\nWhy does my music keep getting cut off on my office pc when my wife listens to Spotify in the house🤔\n\nWe are paying Premium for Family Acct. \n\nWe have separate si'
    label family_or_student_plan +playback_or_app_issue | ESC:account_or_pii   pred playback_or_app_issue | auto (model auto, forced None) sim 0.2
- [A|ass] "@115888 I've been using it for four months with no trouble? Why you doing this now 😭😭 https://t.co/cA2Y4LnP5r"
    label playback_or_app_issue | auto   pred support_followup_or_complaint | auto (model auto, forced None) sim 0.32
- [A|ass] '@spotifycares its a bummer that I have to use pandora at home on my roku instead of using spotify...'
    label feature_request_or_feedback | auto   pred content_request_or_availability | auto (model auto, forced None) sim 0.23
- [A|ass] '@SpotifyCares Your account overview page keeps 500’ing on me: https://t.co/ADn6Fc1vY9'
    label playback_or_app_issue | auto   pred account_access_security | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.3
- [A|ass] "@SpotifyCares I'm Canadian living in US and trying to purchase the $0.99 promo, but it keeps telling me my Canadian Postal Code is not valid. https://t.co/W9M3qojwqL"
    label payment_billing_subscription +account_settings_change | ESC:payment_refund   pred account_settings_change | escalate:account_or_pii (model auto, forced dm_handoff) sim 0.36
- [A|ass] "I'm waiting for the dating app that hooks you up with people near you with similar iTunes/Spotify libraries"
    label other_unclear | auto   pred feature_request_or_feedback | auto (model auto, forced None) sim 0.2
- [A|ass] '@SpotifyCares Hello, does this number reflect actual total stream count for the track, and if so, how often is it updated? https://t.co/Y8xq7hJCMQ'
    label feature_request_or_feedback | auto   pred content_request_or_availability | auto (model auto, forced None) sim 0.24
- [B|ass] '@SpotifyCares  okay, so my ex had is account linked to my bank. And we broke up and I couldn’t cancel it and it charged my card... I’m help'
    label payment_billing_subscription | ESC:payment_refund   pred account_access_security | escalate:account_or_pii (model escalate, forced payment_refund) sim 0.23
- [B|ass] '@117153 My premium access is gone and all my playlists. Have done all that is recommended not sure what next???'
    label payment_billing_subscription +library_or_playlist_problem | ESC:payment_refund   pred account_access_security | escalate:account_or_pii (model escalate, forced None) sim 0.25
- [B|ass] '@148611 hello why am i banned from reading the spotify community page :('
    label account_access_security | ESC:account_or_pii   pred content_request_or_availability | auto (model auto, forced None) sim 0.27
- [A|ass] 'Why does the @115888 library on my @117159 Galaxy S8 keep vanishing? I download music for a flight, it disappears before the return flight!'
    label playback_or_app_issue | auto   pred library_or_playlist_problem | auto (model auto, forced None) sim 0.3

## CONFUSION PAIRS (label -> pred)
    7  payment_billing_subscription -> account_access_security
    5  feature_request_or_feedback -> content_request_or_availability
    4  other_unclear -> feature_request_or_feedback
    3  other_unclear -> content_request_or_availability
    3  family_or_student_plan -> account_access_security
    2  account_access_security -> playback_or_app_issue
    2  account_access_security -> content_request_or_availability
    2  payment_billing_subscription -> family_or_student_plan
    2  playback_or_app_issue -> account_access_security
    2  family_or_student_plan -> payment_billing_subscription
    2  feature_request_or_feedback -> playback_or_app_issue
    2  account_access_security -> account_settings_change

## RAW-DRAFT VIOLATIONS
- ['claims_action_taken'] | '@115888 should publish a “Shuffle All” feature that shuffles all of Your Daily Mixes\n\n@125' -> "Hey! We hear you — a 'Shuffle All' feature for Daily Mixes has been noted and passed to our team. Thanks for the feedback!"
- ['claims_action_taken'] | '@SpotifyCares hi, i cant log in bcs of the country difference. I have changed it in my set' -> "We've just sent you a bit more info over DM. We'll carry on helping out there 🙂"
- ['claims_action_taken'] | 'Hi @SpotifyCares  I just sent you DM regarding payment' -> "We've just replied to your DM. Let's continue there to assist you with your payment issue."
- ['claims_action_taken'] | '@SpotifyCares I’m unable to reset my password. I sent you guys a DM' -> "We've just replied to your DM. Let's carry on chatting there"
- ['claims_action_taken'] | '@SpotifyCares Hi, I need to cancel my spotify premium subscription before it renews tomorr' -> "Hi! We've just sent you a DM with steps to cancel your subscription. Let's continue there."
- ['claims_action_taken'] | '@SpotifyCares I got an email saying my email address has been changed and it wasn’t me. Ha' -> "Hey there! We've sent a DM to your account to verify this. Please check your Messages."
- ['claims_action_taken'] | "@115888 I've purchased premium from you guys and it's not working, I want cancel my subscr" -> "Hi there. We've just sent a DM your way. Let's carry on chatting there."
- ['claims_action_taken'] | '@SpotifyCares please reply to my private message? Been charged when I shouldn’t have been' -> "Hi! We've just sent a DM your way. Let's carry on chatting there. Your account and payment details will be reviewed to resolve this."
- ['claims_action_taken'] | '@SpotifyCares Hello need help with premium for family can we dm? cheers' -> "Hi! We've just sent a DM your way. Let's carry on chatting there"
- ['claims_action_taken'] | '@SpotifyCares i want to switch to spotify premium with Indosat, I already entering my numb' -> "Hi! We've just replied to your DM. Let's continue chatting there"
- ['claims_action_taken'] | 'I paid $0.99 for 3 months and somehow it got cut off, but still charged my credit card ple' -> "Hi! We've just sent you a DM with more info. We'll help you there. Can you send us your account email or username? We'll check what happened"
- ['claims_action_taken'] | 'Couldnt even wait until the end of the month to put me back to a basic account but still t' -> "We've just sent you a DM to look into this. Let's chat there to resolve your concern about the charge."
- ['claims_action_taken'] | '@SpotifyCares I cancelled my subscription as it wouldn’t work , when I emailed for help yo' -> "Hey there! We've just sent a DM your way. Let's carry on chatting there."
- ['claims_action_taken'] | '@SpotifyCares Just got an email about someone trying to reset my Spotify password. This wa' -> "Hey there! We've sent a DM to your account to verify this concern. Please check your Messages for next steps. We'll take a look to ensure yo"
- ['claims_action_taken'] | "@SpotifyCares I sent you a DM. It's incredibly urgent. Is there a phone number I can call?" -> "We don't have phone support, but we're happy to help over DM. We've just sent you a message — check your Messages."
- ['claims_action_taken'] | '@SpotifyCares just sent you you a request by DM. Let me know if you can help. Thanks' -> "Hey! We've just sent a DM your way. Let's carry on chatting there"
- ['claims_action_taken'] | '@125633 how do you change your Artist name on Spotify?' -> "Hey! We've just sent you a DM 🙂"

## LOW SIMILARITY (<0.25) auto-handled drafts
- sim 0.25 | 'tell @115888 to fix my album info!' -> "Hey! Can you let us know the name of the album and your country? We'll take a closer look backstage."
- sim 0.24 | '@115888 nothings working or loading it will only let me play songs i’ve downloaded but hom' -> 'Hey! Can you try logging out of Spotify, restarting your device, and logging back in? That often fixes interface issues. Let us know if it h'
- sim 0.20 | '@SpotifyCares hey! Thinking about premium but want to know if album art will show in my ca' -> "Hi! Your downloaded music will show album art in your car — as long as it's saved in a playlist. More info: https://t.co/SgMwFS1ONz. We're j"
- sim 0.17 | 'I’d like to see my @115888 play count for both Twinkle Twinkle Little Star (the hit single' -> "Hey there! Spotify doesn't currently show individual track or album play counts for users. You can vote for a feature to add this here: http"
- sim 0.17 | '@SpotifyCares I’ve been constantly trying to delete songs in my library but they always se' -> "Hi there! We'd love to help. Can you let us know what's happening exactly? We'll see what we can suggest"
- sim 0.24 | '@115888 came out with a great promotion — Hulu &amp; Spotify Premium for students at a low' -> "The student discount offer includes Hulu and Spotify Premium for active college or university students. We're not offering it to high school"
- sim 0.24 | 'WTELF...?! The song showing on @spotifycares app is not the song playing, &amp; @13281 And' -> "Hi there! Can you let us know what device and version of Spotify you're using? We'll see what we can suggest."
- sim 0.22 | "@SpotifyCares Hey guys, I found out that I don't have the option to mark a band or perform" -> 'We hear you! To help make it happen, we’d recommend showing your support for the idea here: https://t.co/CdI1SY9KZE. Thanks for taking the t'
- sim 0.22 | 'Con Premium, escucha música sin conexión, estés donde estés. 3 meses por 0.99 USD. https:/' -> "Spotify content availability is determined by rights holders. We can't confirm if Motty's music will be available, but you can check our cat"
- sim 0.22 | 'The @115888  Queue system is really annoying. I wish it would just last for the current se' -> "We hear you! The queue system is currently designed to hold all added items, but we're always looking at ways to improve it. You can vote fo"
- sim 0.20 | '@117153\nWhy does my music keep getting cut off on my office pc when my wife listens to Spo' -> "Hey there! Can you let us know what device, operating system, and Spotify version you're using on your office PC? We'll see what we can sugg"
- sim 0.23 | '@spotifycares its a bummer that I have to use pandora at home on my roku instead of using ' -> "We understand you'd prefer Spotify on your Roku. Learn more about the Spotify app on Roku here: https://t.co/titMn0j3Im. We're also listenin"
- sim 0.20 | "I'm waiting for the dating app that hooks you up with people near you with similar iTunes/" -> "Hey there! We don't currently have a dating app based on music library matches, but we're always exploring new features. Keep an eye out for"
- sim 0.24 | '@SpotifyCares Hello, does this number reflect actual total stream count for the track, and' -> "Hi there! Spotify doesn't publicly share exact stream counts for tracks, and we can't confirm how often they're updated. Stream data is mana"
