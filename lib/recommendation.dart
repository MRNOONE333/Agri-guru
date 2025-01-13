import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter/material.dart';
import 'package:flutter1/main.dart';
import 'package:flutter1/tempo.dart';
import 'package:flutter1/crop.dart';
import 'package:flutter1/ferti.dart';
import 'package:get/get.dart';

void main() {
  runApp(
    const MaterialApp(
      title: 'AgriGuru',
      home: Tabs(),
    ),
  );
}

class Tabs extends StatefulWidget {
  const Tabs();
  @override
  TabsState createState() => TabsState();
}

class TabsState extends State<Tabs> {
  @override
  Widget build(BuildContext context) {
    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    return DefaultTabController(
      length: 2,
      child: Scaffold(
          appBar: AppBar(
            toolbarHeight: displayHeight * 0.125,
            title: Center(
              child: Padding(
                padding: EdgeInsets.all(displayWidth * 0.01),
                child: Row(
                  children: [
                    Image.asset(
                      'assets/images/logo.png',
                      width: displayWidth * 0.3,
                    ),
                    SizedBox(
                      width: displayWidth * 0.425,
                    ),
                    TextButton(
                      onPressed: () {},
                      child: const CircleAvatar(
                        backgroundImage: AssetImage('assets/images/avatar.jpg'),
                        radius: 20,
                      ),
                    )
                  ],
                ),
              ),
            ),
            backgroundColor: const Color(0xFFFBFCFA),
            elevation: 0,
            automaticallyImplyLeading: false,
          ),
          body: Column(
            children: [
              SizedBox(
                height: displayHeight * 0.02,
              ),
              Container(
                height: 45,
                padding: EdgeInsets.fromLTRB(3, 5, 3, 5),
                margin: EdgeInsets.fromLTRB(20, 0, 20, 0),
                decoration: BoxDecoration(
                    color: Color(0xffECEEE6),
                    borderRadius: BorderRadius.circular(8)),
                child: TabBar(
                    indicator: BoxDecoration(
                        color: Color(0xffE59A54),
                        borderRadius: BorderRadius.circular(8)),
                    labelColor: Colors.white,
                    unselectedLabelColor: Colors.black,
                    tabs: [
                      Tab(
                        child: Text("crop".tr),
                      ),
                      Tab(text: "fertilizer".tr),
                    ]),
              ),
              const Expanded(
                child: TabBarView(
                  children: [
                    Recommendation(),
                    MyApp(),
                  ],
                ),
              )
            ],
          )),
    );
  }
}