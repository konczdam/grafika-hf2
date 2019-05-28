//=============================================================================================
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Koncz Adam
// Neptun : MOENI1
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h";


// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          
	layout(location = 0) in vec2 cCamWindowVertex;	
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		bool rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};

	struct Ellipsoid {
		vec3 center;
		float radius;
		float radiusX;
		float radiusY;
		float radiusZ;
	};

	struct Triangle{
		vec3 p1,p2,p3;
		vec3 n;
	};

	const int nMaxTriangles = 40;
	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[5];  
	uniform int nObjects;
	uniform Ellipsoid objects[10];
	uniform int nTriangles;
	uniform int mirrorType;
	uniform Triangle triangles[nMaxTriangles];

	in  vec3 p;					
	out vec4 fragmentColor;		

	Hit intersect(const Ellipsoid object, const Ray ray) {
		float radiusZ = object.radiusZ;
		float radiusX = object.radiusX;
		float radiusY = object.radiusY;
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - object.center;
		
		vec3 dir = normalize(ray.dir);
		float a = ((dir.x * dir.x) / (radiusX*radiusX) +
				 (dir.y*dir.y)    / (radiusY*radiusY)	+
				 (dir.z*dir.z)    / (radiusZ*radiusZ));
		float b = ((2.0 * dir.x*dist.x) / (radiusX*radiusX) +
				(2.0 * dir.y*dist.y) / (radiusY*radiusY) +
					(2.0 * dir.z*dist.z) / (radiusZ*radiusZ));
		float c = ((dist.x*dist.x) / (radiusX*radiusX) +
				(dist.y*dist.y) / (radiusY*radiusY)  +
				(dist.z*dist.z) / (radiusZ*radiusZ))  - 1.0f;


		float discr = b * b - 4.0 * a * c;
		if (discr < 0)
		   return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		if (t1 <= 0)
		   return hit;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 intersect = ray.start + dir*hit.t;
		vec3 normal = intersect - object.center;
		hit.normal.x = 2.0 * normal.x / (radiusX * radiusX);
		hit.normal.y = 2.0 * normal.y / (radiusY * radiusY);
		hit.normal.z = 2.0 * normal.z / (radiusZ * radiusZ);
		hit.normal = normalize(hit.normal);
		return hit;
	}

	Hit intersectWithTriangle(const Triangle triangle, const Ray ray){
		Hit hit;
		hit.mat = mirrorType;
		hit.t = -1.0;
		hit.t = dot((triangle.p1 - ray.start), triangle.n) / dot(ray.dir, triangle.n);
		if(hit.t < 0.0)
			return hit;
		hit.position  = ray.start + ray.dir * hit.t;
		hit.normal = triangle.n;
		if(dot(cross((triangle.p2 - triangle.p1) ,(hit.position - triangle.p1)), triangle.n) > 0.0 && 
		   dot(cross((triangle.p3 - triangle.p2) ,(hit.position - triangle.p2)), triangle.n) > 0.0 &&
		   dot(cross((triangle.p1 - triangle.p3) ,(hit.position - triangle.p3)), triangle.n) > 0.0)
			return hit;
		hit.t = -2.0;
		return hit;
}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nObjects; o++) {
			Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
		    hit.mat = o + 2;	 
		
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  
				bestHit = hit;
		}
			
		for(int i = 0; i < nTriangles; i++){
			Hit hit = intersectWithTriangle(triangles[i], ray);
						if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  
								bestHit = hit;
		}			

		if (dot(ray.dir, bestHit.normal) > 0) 
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
		}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int o = 0; o < nObjects; o++) 
			if (intersect(objects[o], ray).t > 0)
			   return true; //  hit.t < 0 if no intersection
		 for (int o = 0; o < nTriangles; o++) {
			if(intersectWithTriangle(triangles[o], ray).t > 0)
				return true;
		}
			return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 16;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0.0, 0.0, 0.0);

		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) {
				outRadiance += weight *  light.La;
				break;
				}
			if (materials[hit.mat].rough) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
					 outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
				break;
			}

			if (materials[hit.mat].reflective) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} 
		}
		return outRadiance;
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) {
		ka = _ka;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}

};

class SmoothMaterial : public Material {
public:
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

vec3 calculateF0(vec3 n, vec3 k) {
	vec3 res;
	res.x = (powf(n.x - 1.0f, 2.0f) + powf(k.x, 2.0f)) / (powf(n.x + 1.0f, 2.0f) + powf(k.x, 2.0f));
	res.y = (powf(n.y - 1.0f, 2.0f) + powf(k.y, 2.0f)) / (powf(n.y + 1.0f, 2.0f) + powf(k.y, 2.0f));
	res.z = (powf(n.z - 1.0f, 2.0f) + powf(k.z, 2.0f)) / (powf(n.z + 1.0f, 2.0f) + powf(k.z, 2.0f));
	return res;
}

SmoothMaterial* Gold() {
	vec3 F0 = calculateF0(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
	return new SmoothMaterial(F0);
}

SmoothMaterial* Silver() {
	vec3 F0 = calculateF0(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
	return new SmoothMaterial(F0);
}

enum mirrorType {
	GOLD = 0,
	SILVER = 1
};

struct Ellipsoid {
	vec3 center;
	float radiusX,radiusY,radiusZ;

	Ellipsoid(const vec3& _center, float _radiusX, float _radiusY, float _radiusZ) { 
		center = _center; 
		radiusX = _radiusX;
		radiusY = _radiusY;
		radiusZ = _radiusZ;
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "objects[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "objects[%d].radiusX", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) 
			glUniform1f(location, radiusX);
		else 
			printf("uniform %s cannot be set\n", buffer);

		sprintf(buffer, "objects[%d].radiusY", o);
		 location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0)
			glUniform1f(location, radiusY);
		else
			printf("uniform %s cannot be set\n", buffer);

		sprintf(buffer, "objects[%d].radiusZ", o);
		 location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0)
			glUniform1f(location, radiusZ);
		else
			printf("uniform %s cannot be set\n", buffer);
	}
};

struct Triangle {
	vec3 p1, p2, p3;
	vec3 n;

	Triangle(const vec3& _p1, const vec3& _p2, const vec3& _p3) {
		p1 = _p1; p2 = _p2; p3 = _p3;
		n = normalize(cross((p2 - p1), (p3 - p1)));
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "triangles[%d].p1", o);
		p1.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].p2", o);
		p2.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].p3", o);
		p3.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].n", o);
		n.SetUniform(shaderProg, buffer);
	}
};

void setMirrorMaterial(mirrorType asd, unsigned int shaderProg) {
	int location = glGetUniformLocation(shaderProg, "mirrorType");
	if (location >= 0)
		glUniform1i(location, asd);
	else
		printf("uniform mirrorType cannot be set\n");
}


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	vec3 getEye() {
		return eye;
	}

	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}

};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};


float rnd(){ 
	return (float)rand() / RAND_MAX; 
}

int rndsign(){ 
	if (rnd() > 0.5f)
		return 1;
	return -1; 
}
GPUProgram gpuProgram;

class Scene {
	std::vector<Ellipsoid*> objects;
	std::vector<Light*> lights;
	std::vector<Material*> materials;
	Camera camera;
public:
	Camera getCamera() {
		return camera;
	}
	void Animate(float dt) {
		camera.Animate(dt);
	}
	void build() {
		setCamera();
		addLigths();
		addMirrorMaterials();
		addRoughMaterials();
		addEllipsoids();
		setMirrorMaterial(GOLD, gpuProgram.getId());
	}
private:	
	void setCamera(){
		vec3 eye = vec3(0.5f, 0.5f, 8.0f);
		vec3 vup = vec3(0.0f, 1.0f, 0.0f);
		vec3 lookat = vec3(0.5f, 0.5f, 0.0f);
		float fov = 45.0f * (float)M_PI / 180.0f;
		camera.set(eye, lookat, vup, fov);
	}
	void addLigths(){
		lights.push_back(new Light(vec3(0.0f, 2.0f, 1.0f), vec3(1.0f, 1.0f, 1.0f), vec3(1.0f, 1.0f, 1.0f)));
	}
	void addMirrorMaterials(){
		materials.push_back(Gold());
		materials.push_back(Silver());
	}
	void addRoughMaterials(){
		materials.push_back(new RoughMaterial(vec3(0.6f, 0.6f, 0.4f), vec3(0.1f, 0.8f, 0.3f), vec3(1.0f, 0.5f, 0.5f), 100.0f));
		materials.push_back(new RoughMaterial(vec3(0.7f, 0.5f, 0.3f), vec3(0.8f, 0.1f, 0.3f), vec3(0.5f, 1.0f, 0.5f), 100.0f));
		materials.push_back(new RoughMaterial(vec3(0.5f, 0.5f, 0.8f), vec3(0.2f, 0.1f, 0.95f), vec3(0.5f, 0.5f, 1.0f), 200.0f));
	}
	void addEllipsoids(){
		objects.push_back(new Ellipsoid(vec3(0.35f, 0.44f, 0.0f), 0.1f, 0.2f, 0.1f));
		objects.push_back(new Ellipsoid(vec3(0.66f, 0.44f, 0.0f), 0.15, 0.1f, 0.1f));
		objects.push_back(new Ellipsoid(vec3(0.505f, 0.69f, 0.0f), 0.12f, 0.14f, 0.1f));
	}

public:
	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nObjects");
		if (location >= 0) 
			glUniform1i(location, objects.size());
		else 
			printf("uniform nObjects cannot be set\n");

		for (unsigned int i = 0; i < objects.size(); i++)
			objects[i]->SetUniform(shaderProg, i);

		lights[0]->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);

		for (unsigned int mat = 0; mat < materials.size(); mat++)
			materials[mat]->SetUniform(shaderProg, mat);
	}


	void doBrown(const unsigned int shaderProg) {
		const float d = 200.0f;
		for (unsigned int i = 0; i < objects.size(); i++) {
			Ellipsoid* tmp = objects[i];
			const vec3 randomvec = vec3(rndsign() * rnd() / d, rndsign() * rnd() / d, 0);
			if (validMove(tmp, randomvec)) {
				tmp->center = tmp->center + randomvec;
				char buffer[128];
				sprintf(buffer, "objects[%d].center", i);
				tmp->center.SetUniform(shaderProg, buffer);
			}
		}
	}
private:
	bool validMove(const Ellipsoid* tmp, const vec3 randomvec) {
		vec3 newcentre = tmp->center + randomvec;
		for(unsigned int i = 0; i < objects.size(); i++)
		if (isInsideBoundaris(newcentre))
			return true;
		return false;
	}

	bool isInsideBoundaris(vec3 centre) {
		return ((centre.x < 0.8f && centre.x >  0.05f) &&
			(centre.y < 1.0f && centre.y >  0.05f));
	}
};

Scene scene;

class MirrorSystemManager {
	int n = 3;
	float rotation = 11.0f * (float)M_PI / 6.0f;
	const int minN = 3;
	const int maxN = 20;
	const float r = 0.55f;
	std::vector<Triangle*> triangles;
	const float rotationScale = 2 * (float)M_PI / 360.0f;
	void deleteTriangles() {
		if (!triangles.empty()) {
			for (unsigned int i = 0; i < triangles.size(); i++)
				delete triangles[i];
			triangles.clear();
		}
	}

public:
	void build() {
		deleteTriangles();
		const float z0 = 0.5f;
		const float z1 = scene.getCamera().getEye().z;
		const vec2 centre = vec2(0.5f, 0.5f);
		const float offset =  (2.0f * (float)M_PI / (float)n);
		for (int i = 0; i < n; i++) {
			const float x0 = cosf(rotation + i*offset)*r + centre.x;
			const float x1 = x0;
			const float x2 = cosf(rotation + (i+1)*offset)*r + centre.x;
			const float y0 = sinf(rotation + i*offset)*r + centre.y;
			const float y1 = y0;
			const float y2 = sinf(rotation + (i + 1)*offset)*r + centre.y;
			triangles.push_back(new Triangle(vec3(x0, y0, z0), vec3(x1, y1, z1), vec3(x2, y2, z1)));
			triangles.push_back(new Triangle(vec3(x2, y2, z0), vec3(x0, y0, z0), vec3(x2, y2, z1)));
		}

	}
	void increaseNumberOfMirrors() {
		if (n < maxN) {
			n++;
			build();
		}
	}

	void decreaseNumberOfMirrors() {
		if (n > minN) {
			n--;
			build();
		}
	}

	void rotateRight() {
		rotation += 2 * (float)M_PI / 360.0f;
		if (rotation >= 2 * (float)M_PI)
			rotation = 0;
		build();
	}

	void rotateLeft() {
		rotation -= 2 * (float)M_PI / 360.0f;
		if (rotation <= 0)
			rotation = 2 * (float)M_PI;
		build();
	}

	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nTriangles");
		if (location >= 0)
			glUniform1i(location, triangles.size());
		else
			printf("uniform nTriangles cannot be set\n");

		for(unsigned int i = 0; i < triangles.size(); i++)
			triangles[i]->SetUniform(shaderProg, i);
	}
};



class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;
MirrorSystemManager mrs;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	scene.build();
	mrs.build();
	fullScreenTexturedQuad.Create();
	gpuProgram.Use();
}


void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.SetUniform(gpuProgram.getId());
	mrs.SetUniform(gpuProgram.getId());
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();								
}

bool rotate = false;
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'r') {
		rotate = !rotate;
	}
	if (key == 'a') {
		mrs.increaseNumberOfMirrors();
	}
	if (key == 'd') {
		mrs.decreaseNumberOfMirrors();
	}
	if (key == 'g') {
		setMirrorMaterial(GOLD, gpuProgram.getId());
	}
	if (key == 's') {
		setMirrorMaterial(SILVER, gpuProgram.getId());
	}
	if (key == '6') { 
		mrs.rotateRight();
	}
	if (key == '4') { 
		mrs.rotateLeft();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	if (rotate)
		scene.Animate(0.01f);
	scene.doBrown(gpuProgram.getId());
	glutPostRedisplay();
}
