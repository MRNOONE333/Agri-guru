import 'package:flutter/material.dart';
import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter1/Lab.dart';
import 'package:flutter1/accountinfo.dart';
import 'package:flutter1/main.dart';
import 'package:flutter1/notific.dart';
import 'package:flutter1/pin.dart';
import 'package:flutter1/recommendation.dart';
import 'package:get/get.dart';
import 'package:flutter1/profile.dart';
import 'package:flutter1/login.dart';
import 'package:firebase_auth/firebase_auth.dart';

class SettingsPage extends StatefulWidget {

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  var currentIndex = 0;
  bool light2 = false;
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          elevation: 0,
          title: Container(
            decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(width: 3.0, color: Color(0xff1B9C73)),
                )
            ),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(0, 0, 0, 8),
              child: Text(
                'setting'.tr,
                style: TextStyle(color: Color(0xff000000)),
              ),
            ),
          ),
          centerTitle: true,
          backgroundColor: Color(0xffffffff),
        ),
        body: Container(
          child: ListView(
            children: [
              // Account Settings
              Container(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8,0,0,8),
                    child: Text('app_set'.tr,style: TextStyle(color: Colors.grey),),
                  )
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.person,color: Color(0xff1C9C73),),
                title: GestureDetector(
                    onTap: () {
                      Navigator.push(context, MaterialPageRoute(builder: (BuildContext context) {
                        return EditProfilePage();
                      })
                      );
                    },
                    child: Text('prof'.tr)),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.info,color: Color(0xff1C9C73),),
                title: GestureDetector(
                    onTap: () {
                      Navigator.push(context, MaterialPageRoute(builder: (BuildContext context) {
                        return AccountInfoPage(name: 'abc', email: '123@email.com', dob: '00/00/00');
                      })
                      );
                    },
                    child: Text('acc_info'.tr)),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),

              SizedBox(height: 15),

              // Security Settings
              Container(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8,0,0,8),
                    child: Text('sec_set'.tr,style: TextStyle(color: Colors.grey),),
                  )
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.lock_open,color: Color(0xff1C9C73),),
                title: GestureDetector(
                    onTap: () {
                      Navigator.push(context, MaterialPageRoute(builder: (BuildContext context) {
                        return PinPage();
                      })
                      );
                    },
                    child: Text('pin'.tr)),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),

              SizedBox(height: 15),

              // App Settings
              Container(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8,5,0,8),
                    child: Text('app_set'.tr,style: TextStyle(color: Colors.grey),),
                  )
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.notifications,color: Color(0xff1C9C73),),
                title: GestureDetector(
                    onTap: () {
                      Navigator.push(context, MaterialPageRoute(builder: (BuildContext context) {
                        return NotificationsPage();
                      })
                      );
                    },
                    child: Text('notif'.tr)),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.language,color: Color(0xff1C9C73),),
                title: Text('hindi'.tr),
                trailing: Switch(
                  value: light2,
                  activeColor:Color(0xff1C9C73),
                  inactiveThumbColor: Color(0xff1C9C73),
                  onChanged: (bool value) {
                    setState(() {
                      light2 = value;
                      if(light2==false){
                        var locale = Locale('en','US');
                        Get.updateLocale(locale);
                      }
                      if(light2==true){
                        var locale = Locale('hi','IN');
                        Get.updateLocale(locale);
                      }
                    });
                  },
                ),
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.share,color: Color(0xff1C9C73),),
                title:  Text('ref_earn'.tr),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),

              SizedBox(height: 15),

              // General
              Container(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8,5,0,8),
                    child: Text('general'.tr,style: TextStyle(color: Colors.grey),),
                  )
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.help,color: Color(0xff1C9C73),),
                title:  Text('faq'.tr),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),
              ListTile(
                tileColor: Color(0xffECEEE6),
                leading: Icon(Icons.description,color: Color(0xff1C9C73),),
                title: Text('term'.tr),
                trailing: Icon(Icons.arrow_forward,color: Color(0xff1C9C73),),
              ),

              SizedBox(height: 20),

              // Logout Button
              Container(
                width: 20,
                margin: EdgeInsets.fromLTRB(52, 0, 52, 20),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(15),
                  color: Colors.green,
                ),
                child: ListTile(
                  title: Center(
                    child: GestureDetector(
                      onTap: (){
                        FirebaseAuth.instance.signOut().then((value) {
                          print("Successfull Logged Out");
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const Login()),
                          );
                        });
                      },
                      child: Text(
                        'log_out'.tr,
                        style: TextStyle(
                          color: Colors.white,

                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),

        // Task Bar
        extendBody: true,
        bottomNavigationBar: FloatingNavigationBar(
            backgroundColor: const Color(0xffD9D9D9),
            iconColor: Colors.black,
            barHeight: 70,
            textStyle: const TextStyle(
              color: Colors.black,
              fontSize: 14.0,
            ),
            indicatorColor: const Color(0xff1C9C73),
            indicatorWidth: 30,
            iconSize: 25.0,
            items: [
              NavBarItems(icon: Icons.home,title: "Nav_home".tr),
              NavBarItems(icon: Icons.addchart,title: "Nav_crop".tr),
              NavBarItems(icon: Icons.location_on, title: "Nav_lab".tr),
              NavBarItems(icon: Icons.settings_rounded,title: "Nav_setting".tr),
            ],
            onChanged: (value) {
              setState(() {
                currentIndex = value;
              });
              if(value==0){
                Navigator.push(context,
                    MaterialPageRoute(builder: (context)=>const HomePage())
                );
              }
              if(value==2){
                Navigator.push(context,
                    MaterialPageRoute(builder: (context)=>const Labpage())
                );
              }
              if(value==3){
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) =>  SettingsPage()),
                );
              }
            }
        ),
      ),
    );
  }
}

void main() {
  runApp(SettingsPage());
}