// import 'dart:html';

import 'package:flutter/material.dart';
import 'package:flutter1/landingPage/styles/button.dart';
import 'package:flutter1/landingPage/styles/color.dart';
import 'package:flutter1/landingPage/styles/typography.dart';
import 'package:flutter1/login.dart';
import 'package:flutter1/signUp.dart';
import 'package:get/get.dart';


class GetStartedPage extends StatefulWidget {
  @override
  State<GetStartedPage> createState() => _GetStartedPageState();
}

class _GetStartedPageState extends State<GetStartedPage> {
  bool light = true;


  @override
  Widget build(BuildContext context) {
    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    return Container(
      decoration: const BoxDecoration(
        image: DecorationImage(
          image: AssetImage('assets/images/bgImage.jpg'),
          fit: BoxFit.cover,
        ),
      ),
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: 24,
            ),
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 50),
                  child: Image.asset('assets/images/logo.png'),
                ),
                SizedBox(
                  height: displayHeight * 0.25,
                ),
                Container(
                  height: 294,
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: white,
                    borderRadius: const BorderRadius.all(
                      Radius.circular(50),
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 24),
                    child: Column(
                      children: [
                        Text(
                          'Start_tog'.tr,
                          style: headerOne,
                        ),
                         SizedBox(
                          height: displayHeight * 0.025,
                        ),
                        Text(
                          'welcome_msg'.tr,
                          style: paragraph,
                        ),
                         SizedBox(
                          height:  displayHeight * 0.025,
                        ),
                        ElevatedButton(
                          style: buttonPrimary,
                          onPressed: () {
                            Navigator.push(context,
                              MaterialPageRoute(builder: (context)=> const SignUp(),)
                            );
                          },
                          child: Text(
                            'get_strt'.tr,
                            style: primaryLabel,
                          ),
                        ),
                        SizedBox(
                          height:  displayHeight * 0.015,
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text('hindi'.tr, style: TextStyle(fontSize: 16)),
                            Switch(
                              value: light,
                              inactiveTrackColor: Colors.grey,
                              activeTrackColor: Colors.grey,
                              activeColor: Colors.green,
                              onChanged: (bool value) {
                                setState(() {
                                  light = value;
                                  if(light==false){
                                    var locale = Locale('hi','IN');
                                    Get.updateLocale(locale);
                                  }
                                  if(light==true){
                                    var locale = Locale('en','US');
                                    Get.updateLocale(locale);
                                  }
                                });

                              },
                            ),
                            Text('english'.tr, style: TextStyle(fontSize: 16)),
                          ],
                        ),
                      ],
                    ),
                  ),
                )

              ],
            ),
          ),
        ),
      ),
    );
  }
}