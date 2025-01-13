import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter1/Setting.dart';
import 'package:flutter1/main.dart';
import 'package:flutter1/recommendation.dart';
import 'package:flutter1/tempo.dart';
import 'package:get/get.dart';
import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter1/model/lab_model.dart';

void main() {
  runApp(
    const MaterialApp(
      title: 'AgriGuru',
      home: Labpage(),
    ),
  );
}

class Labpage extends StatefulWidget {
  const Labpage({Key? key}) : super(key: key);

  @override
  State<Labpage> createState() => _LabpageState();
}

class _LabpageState extends State<Labpage> {
  var currentIndex = 0;

  //Dummy List
  List<LabModel> lab_model = [
    LabModel(Lab_name: 'DIVINE TECHNOLOGIES',Lab_address:'105, First floor, Signature global Synera shopping complex, Sector 81, Gurugram, Haryana 122004' ),
    LabModel(Lab_name: 'FARE Labs', Lab_address: 'D18, Infocity 2 Rd, near Honda Chowk, Phase 2, Info Technology Park, Sector 33, Gurugram, Haryana 122001'),
    LabModel(Lab_name: 'Enviromental Testing Lab', Lab_address: "Nandrampur Bass Rd, near Dena Bank, Ram Nagar, Dharuhera, Haryana 123106 "),
    LabModel(Lab_name: 'Widmans Laboratory', Lab_address: "Plot No. 155E, Sector 7, Imt Manesar, Gurugram, Haryana 122050"),
    LabModel(Lab_name: 'ADVANCE INSPECTION & TESTING LAB', Lab_address: "B-119, Block B, Mayapuri Industrial Area Phase I, Mayapuri, New Delhi, Delhi 110064"),
    LabModel(Lab_name: 'Ace Test House', Lab_address: 'Khasra no. 1048/1088, karnal by pass road, near Laxmi Narayan Mandir, Village, Bhalswa Jahangirpuri, New Delhi, 110033' ),
  ];

  late List<LabModel> display_list = List.from(lab_model);

  void updateList(String value){
  //  Function to filter our list
    setState(() {
      display_list= lab_model.where((element) => element.Lab_name!.toLowerCase().contains(value.toLowerCase())).toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    return Scaffold(
      backgroundColor: Color(0xffECEEE6),
      appBar: AppBar(
        toolbarHeight: displayHeight * 0.125,
        title: Center(
          child: Padding(
            padding: EdgeInsets.all(displayWidth * 0.01),
            child: Row(
              children: [
                Image.asset('assets/images/logo.png',
                  width: displayWidth*0.3,
                ),
                SizedBox(
                  width: displayWidth*0.425,
                ),
                TextButton(
                  onPressed: (){},
                  child: const CircleAvatar(
                    backgroundImage: AssetImage('assets/images/avatar.jpg'),
                    radius: 25,
                  ),
                )],
            ),
          ),
        ),
        backgroundColor: const Color(0xFFECEEE6),
        elevation: 0,
        automaticallyImplyLeading: false,
      ),

      body: Padding(padding: EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              onChanged: (value)=> updateList(value),
              decoration: InputDecoration(
                filled: true,
                fillColor: Colors.white,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8.0),
                  borderSide: BorderSide.none,
                ),
                hintText: 'labs_search'.tr,
                prefixIcon: Icon(Icons.search),
                prefixIconColor: Colors.black,
              ),
            ),
            SizedBox(
              height: 20.0,
            ),
        Container(
          color: Colors.white,
          child: Column(
            children: [
              Container(
                  decoration: BoxDecoration(
                    border: Border(
                      bottom: BorderSide(width: 3.0, color: Color(0xff1B9C73)),
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(0,10,0,7),
                    child: Text('labs_near'.tr,style: TextStyle(fontSize: 22),),
                  )),
              Padding(
                padding: const EdgeInsets.fromLTRB(6, 8, 4, 10),
                child: Text('worry_dont'.tr,style: TextStyle(fontSize: 14),),
              )
            ],
          ),
        ),
            Expanded(child: ListView.builder(
              itemCount: lab_model.length,
              itemBuilder: (context,index)=>Container(
                color: Colors.white,
                child:
                Padding(
                  padding: const EdgeInsets.all(18.0),
                  child: Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(8),
                      color: Color(0xffECEEE6),
                    ),
                    child:
                    ListTile(
                      contentPadding: EdgeInsets.all(8.0),
                      title: Text(display_list[index].Lab_name!,style: TextStyle(color: Colors.black,fontWeight: FontWeight.bold ),),
                      subtitle: Padding(
                        padding: const EdgeInsets.fromLTRB(0.0,8.0, 0.0, 0.0),
                        child: Text(display_list[index].Lab_address!,style: TextStyle(color:Colors.black,fontWeight: FontWeight.normal),),
                      ),
                      trailing: Icon(CupertinoIcons.lab_flask,color: Color(0xff34A853),),
                    ),
                  ),
                ),
              ),

            ))
          ],

      ),
      ),



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
                MaterialPageRoute(builder: (context) => SettingsPage()),
              );
            }
          }
      ),
    );
  }
}




