import cv2
import os
import numpy as np

test_28 = [

'01d32d72-bacd-11e8-b2b8-ac1f6b6435d0',
'04e82088-bad4-11e8-b2b8-ac1f6b6435d0',
'1bcde1d2-bac7-11e8-b2b7-ac1f6b6435d0',
'1ebb230a-bad1-11e8-b2b8-ac1f6b6435d0',
'3539e7f8-bad4-11e8-b2b8-ac1f6b6435d0',
'3fe1a9f8-bad8-11e8-b2b9-ac1f6b6435d0',
'4104fc8e-bad5-11e8-b2b9-ac1f6b6435d0',
'4422be80-bac9-11e8-b2b8-ac1f6b6435d0',
'590414c8-bad0-11e8-b2b8-ac1f6b6435d0',
'70d9b858-bad7-11e8-b2b9-ac1f6b6435d0',
'745448b6-bac5-11e8-b2b7-ac1f6b6435d0',
'76f171d6-bad2-11e8-b2b8-ac1f6b6435d0',
'795d77fe-bacc-11e8-b2b8-ac1f6b6435d0',
'86c164e6-bac7-11e8-b2b7-ac1f6b6435d0',
'9103658c-bac5-11e8-b2b7-ac1f6b6435d0',
'92a03f8c-baca-11e8-b2b8-ac1f6b6435d0',
'97989b76-bad2-11e8-b2b8-ac1f6b6435d0',
'a14399ee-bad4-11e8-b2b8-ac1f6b6435d0',
'a1f0873a-bacf-11e8-b2b8-ac1f6b6435d0',
'af2177a0-bacc-11e8-b2b8-ac1f6b6435d0',
'b15e17be-bac5-11e8-b2b7-ac1f6b6435d0',
'ca35a3d0-bad2-11e8-b2b8-ac1f6b6435d0',
'd8e64b1c-baca-11e8-b2b8-ac1f6b6435d0',
'defda53c-bace-11e8-b2b8-ac1f6b6435d0',
'dfd49804-bad7-11e8-b2b9-ac1f6b6435d0',
'fb19471e-bad6-11e8-b2b9-ac1f6b6435d0',
'fd3fdc76-bad4-11e8-b2b8-ac1f6b6435d0'

]

train_28 = [
'05d32f36-bba3-11e8-b2b9-ac1f6b6435d0',
'082a828a-bbbb-11e8-b2ba-ac1f6b6435d0',
'0afda11a-bba0-11e8-b2b9-ac1f6b6435d0',
'18df69fc-bbb5-11e8-b2ba-ac1f6b6435d0',
'2b3ce424-bba8-11e8-b2ba-ac1f6b6435d0',
'43f6bd88-bbc5-11e8-b2bc-ac1f6b6435d0',
'70b97ed2-bbac-11e8-b2ba-ac1f6b6435d0',
'802998d4-bbbb-11e8-b2ba-ac1f6b6435d0',
'b1131086-bb9f-11e8-b2b9-ac1f6b6435d0',
'c9806c74-bbca-11e8-b2bc-ac1f6b6435d0',
'e403806e-bbbf-11e8-b2bb-ac1f6b6435d0'
]

test_16 = [
'04fe60d6-bad5-11e8-b2b8-ac1f6b6435d0',
'06e54e5e-bac7-11e8-b2b7-ac1f6b6435d0',
'0d238c04-bad6-11e8-b2b9-ac1f6b6435d0',
'10748996-baca-11e8-b2b8-ac1f6b6435d0',
'11693760-bada-11e8-b2b9-ac1f6b6435d0',
'30679a8c-bace-11e8-b2b8-ac1f6b6435d0',
'3b2d1274-bacb-11e8-b2b8-ac1f6b6435d0',
'443b81cc-bac9-11e8-b2b8-ac1f6b6435d0',
'46b3ac54-bad8-11e8-b2b9-ac1f6b6435d0',
'56e5eac6-bac7-11e8-b2b7-ac1f6b6435d0',
'5a50c16a-baca-11e8-b2b8-ac1f6b6435d0',
'6742fb2e-bac8-11e8-b2b8-ac1f6b6435d0',
'7fcba676-bad9-11e8-b2b9-ac1f6b6435d0',
'80cd02a6-bacd-11e8-b2b8-ac1f6b6435d0',
'89975d50-bad7-11e8-b2b9-ac1f6b6435d0',
'b5764aca-bace-11e8-b2b8-ac1f6b6435d0',
'c43aea58-bacd-11e8-b2b8-ac1f6b6435d0',
'c583acc0-bacc-11e8-b2b8-ac1f6b6435d0',
'c7109768-bad8-11e8-b2b9-ac1f6b6435d0',
'c7f2fd0c-bad2-11e8-b2b8-ac1f6b6435d0',
'd0812898-bad4-11e8-b2b8-ac1f6b6435d0',
'd342255e-bada-11e8-b2b9-ac1f6b6435d0',
'db3fcdd8-bacb-11e8-b2b8-ac1f6b6435d0',
'de0ed5c2-bad0-11e8-b2b8-ac1f6b6435d0',
'e29dd5f6-bac7-11e8-b2b7-ac1f6b6435d0',
'ee922f9e-bacf-11e8-b2b8-ac1f6b6435d0',
'fdbd4f3a-bac5-11e8-b2b7-ac1f6b6435d0'

]

train_16 = [
'381e477c-bb9a-11e8-b2b9-ac1f6b6435d0',
'4f4433ea-bbad-11e8-b2ba-ac1f6b6435d0',
'58cb3d80-bb9b-11e8-b2b9-ac1f6b6435d0',
'5b18c856-bbc2-11e8-b2bb-ac1f6b6435d0',
'61a51908-bbc8-11e8-b2bc-ac1f6b6435d0',
'68a3f5f4-bba4-11e8-b2b9-ac1f6b6435d0',
'6c21c47e-bbb6-11e8-b2ba-ac1f6b6435d0',
'7666cca6-bbaa-11e8-b2ba-ac1f6b6435d0',
'7ee3439a-bbc9-11e8-b2bc-ac1f6b6435d0',
'80e422c4-bbc7-11e8-b2bc-ac1f6b6435d0',
'85da80ce-bbb9-11e8-b2ba-ac1f6b6435d0',
'8a384340-bbaa-11e8-b2ba-ac1f6b6435d0',
'9515e652-bbb7-11e8-b2ba-ac1f6b6435d0',
'a5d6de0a-bb9a-11e8-b2b9-ac1f6b6435d0',
'a74e60b8-bbc8-11e8-b2bc-ac1f6b6435d0',
'b07d8ed6-bba3-11e8-b2b9-ac1f6b6435d0',
'b0b13c66-bbaa-11e8-b2ba-ac1f6b6435d0',
'c361f992-bbad-11e8-b2ba-ac1f6b6435d0',
'ca4b50fa-bbc1-11e8-b2bb-ac1f6b6435d0',
'f38ad554-bba7-11e8-b2ba-ac1f6b6435d0',
'ff500aee-bba2-11e8-b2b9-ac1f6b6435d0'

]



DEBUG_DIR = 'debug/' + 'test_16'
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def open_rgby(id): #a function that reads RGBY image
    colors = ['blue','green','red','yellow']
    #colors = ['red']
    # colors = ['green']
    # colors = ['blue']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join('../input/test', id+'_'+color+'.png'), flags).astype(np.float32)
           for color in colors]
    return np.stack(img, axis=-1)

for id in test_16:
    img = open_rgby(id)
    cv2.imwrite(DEBUG_DIR + '/' + id + ".jpg", img)
    cv2.imwrite(DEBUG_DIR + '/' + id + '_blue' + ".jpg", img[:,:,0])
    cv2.imwrite(DEBUG_DIR + '/' + id + '_green' + ".jpg", img[:,:,1])
    cv2.imwrite(DEBUG_DIR + '/' + id + '_red' + ".jpg", img[:,:,2])
